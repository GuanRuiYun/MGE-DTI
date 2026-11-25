# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import GatedConv1d


def _make_sequence_mask(lengths, max_len, device):
    lengths = lengths.long().view(-1).clamp(min=0, max=max_len)
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _masked_mean(tokens, mask, dim=1):
    mask = mask.bool()
    denom = mask.float().sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (tokens * mask.unsqueeze(-1)).sum(dim=dim) / denom


def _sparsemax(logits, dim=-1):
    logits = logits.transpose(dim, -1)
    original_shape = logits.shape
    logits = logits.reshape(-1, logits.size(-1))
    logits = logits - logits.max(dim=-1, keepdim=True).values

    zs = torch.sort(logits, dim=-1, descending=True).values
    range_values = torch.arange(1, zs.size(-1) + 1, device=logits.device, dtype=logits.dtype).view(1, -1)
    bound = 1.0 + range_values * zs
    cumulative = zs.cumsum(dim=-1)
    is_active = bound > cumulative
    k = is_active.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (cumulative.gather(1, k - 1) - 1.0) / k.to(logits.dtype)
    output = torch.clamp(logits - tau, min=0.0)

    output = output.reshape(original_shape).transpose(dim, -1)
    return output


def _masked_sparsemax(logits, mask, dim=-1):
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e4)
    probs = _sparsemax(masked_logits, dim=dim) * mask.float()
    denom = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp(min=1e-8), torch.zeros_like(probs))
    return probs


def _masked_softmax(logits, mask, dim=-1):
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e4)
    probs = F.softmax(masked_logits, dim=dim) * mask.float()
    denom = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp(min=1e-8), torch.zeros_like(probs))
    return probs


def _masked_entropy(probs, mask, dim=-1, eps=1e-8):
    probs = probs.clamp(min=eps)
    entropy = -(probs * probs.log()).sum(dim=dim)
    if mask.dim() == entropy.dim() + 1:
        mask = mask.any(dim=-1)
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (entropy * mask).sum() / denom


def _align_token_and_mask(tokens, mask):
    mask = mask.bool()
    token_len = tokens.size(1)
    mask_len = mask.size(1)
    if token_len == mask_len:
        return tokens, mask
    if token_len < mask_len:
        return tokens, mask[:, :token_len]

    pad_len = token_len - mask_len
    pad = torch.zeros(mask.size(0), pad_len, dtype=torch.bool, device=mask.device)
    return tokens, torch.cat([mask, pad], dim=1)


def _resize_token_sequence(tokens, target_len):
    token_len = tokens.size(1)
    if token_len == target_len:
        return tokens
    if token_len > target_len:
        return tokens[:, :target_len]

    pad_shape = (tokens.size(0), target_len - token_len, tokens.size(2))
    pad = torch.zeros(pad_shape, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([tokens, pad], dim=1)


def _resize_score_sequence(scores, target_len):
    score_len = scores.size(1)
    if score_len == target_len:
        return scores
    if score_len > target_len:
        return scores[:, :target_len]

    pad = torch.zeros(
        scores.size(0),
        target_len - score_len,
        dtype=scores.dtype,
        device=scores.device
    )
    return torch.cat([scores, pad], dim=1)


def _select_topk_tokens(tokens, mask, scores, topk):
    tokens, mask = _align_token_and_mask(tokens, mask)
    scores = _resize_score_sequence(scores.to(device=tokens.device, dtype=tokens.dtype), tokens.size(1))
    masked_scores = scores.masked_fill(~mask.bool(), -1e4)
    k = max(1, min(int(topk), tokens.size(1)))
    top_scores, top_indices = torch.topk(masked_scores, k=k, dim=1, largest=True, sorted=False)
    gather_index = top_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    top_tokens = torch.gather(tokens, 1, gather_index)
    top_mask = torch.gather(mask.long(), 1, top_indices).bool()
    top_tokens = top_tokens * top_mask.unsqueeze(-1)
    return top_tokens, top_mask, top_indices, top_scores


class StructureOptionalProteinEncoder(nn.Module):
    """Sequence-first protein encoder with optional latent pocket tokens."""

    def __init__(self, char_dim, conv_dim, out_dim, protein_kernel, num_pocket_slots=16,
                 residue_dropout=0.0, protein_input_mode="esm", esm_dim=1280,
                 esm_num_layers=4, protein_adapter_layers=2,
                 pocket_refine_layers=0, pocket_refine_heads=4, pocket_refine_dropout=0.1):
        super().__init__()
        self.residue_dropout = float(residue_dropout)
        self.protein_input_mode = str(protein_input_mode).lower()
        self.esm_num_layers = max(1, int(esm_num_layers))
        self.protein_embed = None
        self.gate1 = None
        self.gate2 = None
        self.gate3 = None
        self.esm_proj = None
        self.esm_layer_weights = None
        self.esm_adapter = None
        self.raw_residue_proj = None

        if self.protein_input_mode == "esm":
            self.esm_layer_weights = nn.Parameter(torch.zeros(self.esm_num_layers))
            self.esm_proj = nn.Sequential(
                nn.LayerNorm(int(esm_dim)),
                nn.Linear(int(esm_dim), out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim)
            )
            if int(protein_adapter_layers) > 0:
                adapter_layer = nn.TransformerEncoderLayer(
                    d_model=out_dim,
                    nhead=4,
                    dim_feedforward=out_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True
                )
                self.esm_adapter = nn.TransformerEncoder(
                    adapter_layer,
                    num_layers=int(protein_adapter_layers)
                )
        else:
            self.protein_embed = nn.Embedding(26, char_dim, padding_idx=0)
            self.gate1 = GatedConv1d(char_dim, conv_dim, protein_kernel[0], stride=1, padding=0)
            self.gate2 = GatedConv1d(conv_dim, conv_dim * 2, protein_kernel[1], stride=1, padding=0)
            self.gate3 = GatedConv1d(conv_dim * 2, conv_dim * 4, protein_kernel[2], stride=1, padding=0)
            self.raw_residue_proj = nn.Linear(conv_dim * 4, out_dim)

        self.residue_norm = nn.LayerNorm(out_dim)
        self.pocket_queries = nn.Parameter(torch.randn(num_pocket_slots, out_dim) * 0.02)
        self.pocket_attn = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
        self.pocket_norm = nn.LayerNorm(out_dim)
        self.pocket_refiner = None
        if int(pocket_refine_layers) > 0:
            pocket_layer = nn.TransformerEncoderLayer(
                d_model=out_dim,
                nhead=max(1, int(pocket_refine_heads)),
                dim_feedforward=out_dim * 2,
                dropout=float(pocket_refine_dropout),
                activation="gelu",
                batch_first=True
            )
            self.pocket_refiner = nn.TransformerEncoder(
                pocket_layer,
                num_layers=int(pocket_refine_layers)
            )

    def forward(self, protein):
        if not isinstance(protein, dict):
            raise TypeError("Protein input must be a dict-like batch, got {}".format(type(protein)))

        input_type = str(protein.get("input_type", self.protein_input_mode)).lower()
        if input_type == "esm":
            protein_embedding = protein["esm_embedding"]
            if protein_embedding.dim() == 2:
                protein_embedding = protein_embedding.unsqueeze(0).unsqueeze(0)
            elif protein_embedding.dim() == 3:
                protein_embedding = protein_embedding.unsqueeze(1)
            if protein_embedding.dim() != 4:
                raise ValueError("ESM protein embedding must have shape [B, K, L, D], got {}".format(
                    tuple(protein_embedding.shape)
                ))
            num_layers = protein_embedding.size(1)
            if num_layers == 1:
                mixed_embedding = protein_embedding[:, 0]
            else:
                layer_weights = self.esm_layer_weights[:num_layers]
                layer_weights = torch.softmax(layer_weights, dim=0)
                mixed_embedding = torch.einsum("k,bkld->bld", layer_weights, protein_embedding)
            residue = self.residue_norm(self.esm_proj(mixed_embedding))
            lengths = protein.get("esm_lengths")
            if lengths is None:
                residue_mask = torch.ones(
                    residue.size(0), residue.size(1), dtype=torch.bool, device=residue.device
                )
            else:
                residue_mask = _make_sequence_mask(lengths, residue.size(1), residue.device)
            if self.esm_adapter is not None:
                residue = self.esm_adapter(residue, src_key_padding_mask=~residue_mask)
        else:
            raw_tokens = protein["raw_tokens"]
            protein_embed = self.protein_embed(raw_tokens).permute(0, 2, 1).contiguous()
            residue_conv = self.gate3(self.gate2(self.gate1(protein_embed)))
            residue = self.residue_norm(self.raw_residue_proj(residue_conv.permute(0, 2, 1).contiguous()))
            residue_mask = torch.where(raw_tokens > 0, True, False)[:, :residue.size(1)]

        if self.training and self.residue_dropout > 0:
            keep = torch.rand(residue_mask.shape, device=residue.device) > self.residue_dropout
            keep = keep | ~residue_mask
            residue = residue * keep.unsqueeze(-1)

        residue = residue * residue_mask.unsqueeze(-1)

        batch_size = residue.size(0)
        pocket_query = self.pocket_queries.unsqueeze(0).expand(batch_size, -1, -1)
        pocket_tokens, pocket_attention = self.pocket_attn(
            pocket_query,
            residue,
            residue,
            key_padding_mask=~residue_mask,
            need_weights=True,
            average_attn_weights=False
        )
        pocket_tokens = self.pocket_norm(pocket_tokens)
        if self.pocket_refiner is not None:
            pocket_tokens = self.pocket_refiner(pocket_tokens)
            pocket_tokens = self.pocket_norm(pocket_tokens)
        pocket_mask = torch.ones(
            batch_size,
            pocket_tokens.size(1),
            dtype=torch.bool,
            device=residue.device
        )

        candidate_tokens = torch.cat([residue, pocket_tokens], dim=1)
        candidate_mask = torch.cat([residue_mask, pocket_mask], dim=1)
        global_token = _masked_mean(residue, residue_mask)

        return {
            "residue": residue,
            "residue_mask": residue_mask,
            "candidate": candidate_tokens,
            "candidate_mask": candidate_mask,
            "pocket": pocket_tokens,
            "global": global_token,
            "pocket_attention": pocket_attention
        }


class SparseEvidenceAggregator(nn.Module):
    def __init__(self, out_dim, topk=16, dropout=0.1, context_layers=2, context_heads=4,
                 attention_normalizer="sparsemax"):
        super().__init__()
        self.out_dim = out_dim
        self.topk = int(topk)
        self.attention_normalizer = str(attention_normalizer).lower()
        if self.attention_normalizer not in {"sparsemax", "softmax"}:
            raise ValueError("attention_normalizer must be sparsemax or softmax")
        self.motif_proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.ReLU())
        self.residue_proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.ReLU())
        self.pair_proj = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
        )
        self.motif_to_residue_attn = nn.MultiheadAttention(
            out_dim,
            num_heads=max(1, int(context_heads)),
            dropout=dropout,
            batch_first=True
        )
        self.residue_to_motif_attn = nn.MultiheadAttention(
            out_dim,
            num_heads=max(1, int(context_heads)),
            dropout=dropout,
            batch_first=True
        )
        self.motif_attn_norm = nn.LayerNorm(out_dim)
        self.residue_attn_norm = nn.LayerNorm(out_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=max(1, int(context_heads)),
            dim_feedforward=out_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(context_layers))
        )
        self.motif_scorer = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1)
        )
        self.local_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 2)
        )

    def _masked_attention(self, logits, mask, dim=-1):
        if self.attention_normalizer == "softmax":
            return _masked_softmax(logits, mask, dim=dim)
        return _masked_sparsemax(logits, mask, dim=dim)

    def forward(self, motif_tokens, residue_tokens, motif_mask, residue_mask):
        motif_tokens, motif_mask = _align_token_and_mask(motif_tokens, motif_mask)
        residue_tokens, residue_mask = _align_token_and_mask(residue_tokens, residue_mask)
        target_motif_len = motif_mask.size(1)

        motif_tokens = self.motif_proj(motif_tokens)
        residue_tokens = self.residue_proj(residue_tokens)

        motif_cross, _ = self.motif_to_residue_attn(
            motif_tokens,
            residue_tokens,
            residue_tokens,
            key_padding_mask=~residue_mask,
            need_weights=False
        )
        motif_tokens = self.motif_attn_norm(motif_tokens + motif_cross)
        motif_tokens = motif_tokens * motif_mask.unsqueeze(-1)

        safe_motif_mask = motif_mask.clone()
        safe_motif_tokens = motif_tokens
        empty_motifs = ~safe_motif_mask.any(dim=1)
        if empty_motifs.any():
            safe_motif_mask[empty_motifs, 0] = True
            safe_motif_tokens = safe_motif_tokens.clone()
            safe_motif_tokens[empty_motifs, 0] = 0.0
        residue_cross, _ = self.residue_to_motif_attn(
            residue_tokens,
            safe_motif_tokens,
            safe_motif_tokens,
            key_padding_mask=~safe_motif_mask,
            need_weights=False
        )
        residue_tokens = self.residue_attn_norm(residue_tokens + residue_cross)
        residue_tokens = residue_tokens * residue_mask.unsqueeze(-1)

        scores = torch.einsum("bmd,brd->bmr", motif_tokens, residue_tokens) / math.sqrt(self.out_dim)
        valid = motif_mask.unsqueeze(2) & residue_mask.unsqueeze(1)
        residue_attention = self._masked_attention(scores, valid, dim=-1)
        residue_context = torch.einsum("bmr,brd->bmd", residue_attention, residue_tokens)

        pair_features = torch.cat([
            motif_tokens,
            residue_context,
            motif_tokens * residue_context,
            torch.abs(motif_tokens - residue_context)
        ], dim=-1)
        pair_tokens = self.pair_proj(pair_features) * motif_mask.unsqueeze(-1)

        safe_mask = motif_mask.clone()
        empty_samples = ~safe_mask.any(dim=1)
        if empty_samples.any():
            safe_mask[empty_samples, 0] = True
            pair_tokens = pair_tokens.clone()
            pair_tokens[empty_samples, 0] = 0.0

        pair_tokens = self.context_encoder(pair_tokens, src_key_padding_mask=~safe_mask)
        pair_tokens = _resize_token_sequence(pair_tokens, target_motif_len)
        pair_tokens = pair_tokens * motif_mask.unsqueeze(-1)

        motif_logits = self.motif_scorer(pair_tokens).squeeze(-1)
        motif_attention = self._masked_attention(motif_logits, motif_mask, dim=-1)
        evidence = (pair_tokens * motif_attention.unsqueeze(-1)).sum(dim=1)
        local_logits = self.local_head(evidence)

        has_motif = motif_mask.any(dim=1, keepdim=True)
        local_logits = torch.where(has_motif, local_logits, torch.zeros_like(local_logits))
        evidence = torch.where(has_motif, evidence, torch.zeros_like(evidence))

        motif_count = motif_mask.float().sum(dim=1).clamp(min=1.0)
        residue_sparsity = 1.0 - residue_attention.pow(2).sum(dim=-1)
        residue_sparsity = (residue_sparsity * motif_mask.float()).sum(dim=1) / motif_count
        motif_sparsity = 1.0 - motif_attention.pow(2).sum(dim=-1)
        evidence_sparsity_loss = (residue_sparsity + motif_sparsity).mean()

        batch_size, _, num_residues = scores.shape
        flat_scores = scores.view(batch_size, -1)
        flat_valid = valid.view(batch_size, -1)
        flat_scores = flat_scores.masked_fill(~flat_valid, -1e4)
        explain_k = max(1, min(self.topk, flat_scores.size(1)))
        top_scores, top_indices = torch.topk(flat_scores, k=explain_k, dim=-1)
        top_motif_indices = top_indices // num_residues
        top_residue_indices = top_indices % num_residues
        top_scores = torch.where(
            flat_valid.any(dim=-1, keepdim=True),
            top_scores,
            torch.zeros_like(top_scores)
        )
        evidence_norm = evidence.norm(dim=-1).mean()
        pair_score_mean = scores.masked_select(valid).mean() if valid.any() else scores.sum() * 0.0
        motif_entropy = _masked_entropy(motif_attention, motif_mask)
        residue_entropy = _masked_entropy(residue_attention, motif_mask.unsqueeze(-1), dim=-1)

        return {
            "local_logits": local_logits,
            "top_scores": top_scores,
            "top_motif_indices": top_motif_indices,
            "top_residue_indices": top_residue_indices,
            "evidence_sparsity_loss": evidence_sparsity_loss,
            "evidence_norm": evidence_norm,
            "pair_score_mean": pair_score_mean,
            "motif_attention_entropy": motif_entropy,
            "residue_attention_entropy": residue_entropy
        }


class CoarseToFineInteractionHead(nn.Module):
    """Performance-oriented atom-pocket interaction head."""

    def __init__(self, out_dim, dropout=0.1, heads=4, disable_coarse_context=False):
        super().__init__()
        heads = max(1, int(heads))
        self.disable_coarse_context = bool(disable_coarse_context)
        self.atom_norm = nn.LayerNorm(out_dim)
        self.motif_norm = nn.LayerNorm(out_dim)
        self.pocket_norm = nn.LayerNorm(out_dim)
        self.target_norm = nn.LayerNorm(out_dim)
        self.motif_to_pocket = nn.MultiheadAttention(
            out_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.atom_to_target = nn.MultiheadAttention(
            out_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.coarse_norm = nn.LayerNorm(out_dim)
        self.fine_norm = nn.LayerNorm(out_dim)
        self.atom_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(out_dim * 8),
            nn.Linear(out_dim * 8, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 2)
        )

    @staticmethod
    def _safe_key_mask(tokens, mask):
        mask = mask.bool()
        safe_mask = mask.clone()
        empty_samples = ~safe_mask.any(dim=1)
        if empty_samples.any():
            safe_mask[empty_samples, 0] = True
            tokens = tokens.clone()
            tokens[empty_samples, 0] = 0.0
        return tokens, safe_mask

    def forward(self, atom_tokens, atom_mask, motif_tokens, motif_mask,
                pocket_tokens, pocket_mask, target_tokens, target_mask,
                drug_global, protein_global):
        atom_tokens, atom_mask = _align_token_and_mask(atom_tokens, atom_mask)
        motif_tokens, motif_mask = _align_token_and_mask(motif_tokens, motif_mask)
        pocket_tokens, pocket_mask = _align_token_and_mask(pocket_tokens, pocket_mask)
        target_tokens, target_mask = _align_token_and_mask(target_tokens, target_mask)

        atom_tokens = self.atom_norm(atom_tokens) * atom_mask.unsqueeze(-1)
        motif_tokens = self.motif_norm(motif_tokens) * motif_mask.unsqueeze(-1)
        pocket_tokens = self.pocket_norm(pocket_tokens) * pocket_mask.unsqueeze(-1)
        target_tokens = self.target_norm(target_tokens) * target_mask.unsqueeze(-1)

        pocket_tokens, safe_pocket_mask = self._safe_key_mask(pocket_tokens, pocket_mask)
        target_tokens, safe_target_mask = self._safe_key_mask(target_tokens, target_mask)

        coarse_cross, _ = self.motif_to_pocket(
            motif_tokens,
            pocket_tokens,
            pocket_tokens,
            key_padding_mask=~safe_pocket_mask,
            need_weights=False
        )
        coarse_tokens = self.coarse_norm(motif_tokens + coarse_cross) * motif_mask.unsqueeze(-1)
        coarse_summary = _masked_mean(coarse_tokens, motif_mask)
        pocket_summary = _masked_mean(pocket_tokens, safe_pocket_mask)
        has_motif = motif_mask.any(dim=1, keepdim=True)
        coarse_summary = torch.where(has_motif, coarse_summary, pocket_summary)

        coarse_context = coarse_summary.unsqueeze(1).expand(-1, atom_tokens.size(1), -1)
        if self.disable_coarse_context:
            atom_gate = torch.zeros_like(atom_tokens)
            guided_atoms = atom_tokens
        else:
            atom_gate = self.atom_gate(torch.cat([atom_tokens, coarse_context], dim=-1))
            guided_atoms = atom_tokens + atom_gate * coarse_context
        fine_cross, _ = self.atom_to_target(
            guided_atoms,
            target_tokens,
            target_tokens,
            key_padding_mask=~safe_target_mask,
            need_weights=False
        )
        fine_tokens = self.fine_norm(guided_atoms + fine_cross) * atom_mask.unsqueeze(-1)
        fine_summary = _masked_mean(fine_tokens, atom_mask)

        fused_feature = torch.cat([
            fine_summary,
            coarse_summary,
            drug_global,
            protein_global,
            fine_summary * protein_global,
            torch.abs(fine_summary - protein_global),
            fine_summary * coarse_summary,
            torch.abs(fine_summary - coarse_summary)
        ], dim=-1)
        feature = self.feature_proj(fused_feature)
        logits = self.classifier(feature)

        atom_count = atom_mask.float().sum().clamp(min=1.0)
        atom_gate_mean = (atom_gate.squeeze(-1) if atom_gate.size(-1) == 1 else atom_gate.mean(dim=-1))
        atom_gate_mean = (atom_gate_mean * atom_mask.float()).sum() / atom_count

        return {
            "logits": logits,
            "fine_feature_norm": fine_summary.norm(dim=-1).mean(),
            "coarse_feature_norm": coarse_summary.norm(dim=-1).mean(),
            "performance_feature_norm": feature.norm(dim=-1).mean(),
            "atom_pocket_gate_mean": atom_gate_mean,
            "fine_target_count": target_mask.float().sum(dim=1).mean()
        }


class EvidenceDTI(nn.Module):
    """Differentiable sparse-evidence DTI predictor."""

    def __init__(self, args, config, device):
        super().__init__()
        self.device = device
        self.out_dim = config["out_dim"]
        self.global_calibration_max = float(getattr(args, "global_calibration_max", 0.08))
        self.evidence_aux_logit_weight = float(getattr(args, "evidence_aux_logit_weight", 0.0))
        self.fine_interaction_target = str(getattr(args, "fine_interaction_target", "pocket")).lower()
        self.fine_residue_topk = int(getattr(args, "fine_residue_topk", 128))
        self.prediction_mode = str(getattr(args, "prediction_mode", "full")).lower()
        if self.prediction_mode not in {"full", "performance_only", "local_only", "global_only"}:
            raise ValueError("prediction_mode must be full, performance_only, local_only, or global_only")

        self.drug_proj = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU()
        )
        self.fg_proj = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU()
        )
        self.protein_encoder = StructureOptionalProteinEncoder(
            config["char_dim"],
            config["conv"],
            self.out_dim,
            config["protein_kernel"],
            num_pocket_slots=int(getattr(args, "pocket_slots", 16)),
            residue_dropout=float(getattr(args, "residue_dropout", 0.0)),
            protein_input_mode=getattr(args, "protein_input_mode", "esm"),
            esm_dim=int(getattr(args, "esm_embedding_dim", 1280)),
            esm_num_layers=int(getattr(args, "esm_num_layers", 4)),
            protein_adapter_layers=int(getattr(args, "protein_adapter_layers", 2)),
            pocket_refine_layers=int(getattr(args, "pocket_refine_layers", 0)),
            pocket_refine_heads=int(getattr(args, "pocket_refine_heads", 4)),
            pocket_refine_dropout=float(getattr(args, "pocket_refine_dropout", 0.1))
        )
        self.evidence = SparseEvidenceAggregator(
            self.out_dim,
            topk=int(getattr(args, "evidence_topk", 16)),
            dropout=float(config.get("fc_dropout", 0.1)),
            context_layers=int(getattr(args, "evidence_context_layers", 2)),
            context_heads=int(getattr(args, "evidence_context_heads", 4)),
            attention_normalizer=getattr(args, "evidence_attention", "sparsemax")
        )
        self.performance_head = CoarseToFineInteractionHead(
            self.out_dim,
            dropout=float(getattr(args, "performance_interaction_dropout", config.get("fc_dropout", 0.1))),
            heads=int(getattr(args, "performance_interaction_heads", 4)),
            disable_coarse_context=bool(getattr(args, "disable_coarse_context", False))
        )
        self.global_calibrator = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.ReLU(),
            nn.Dropout(float(config.get("fc_dropout", 0.1))),
            nn.Linear(self.out_dim, 2)
        )
        self.global_gate = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.ReLU(),
            nn.Dropout(float(config.get("fc_dropout", 0.1))),
            nn.Linear(self.out_dim, 1)
        )
        self.last_explanations = {}

    def forward(self, drug, fg, num_atoms, protein, fg_mask, return_aux=False):
        drug_mask = _make_sequence_mask(num_atoms, drug.size(1), drug.device)
        drug_tokens = self.drug_proj(drug) * drug_mask.unsqueeze(-1)
        drug_global = _masked_mean(drug_tokens, drug_mask)

        fg_mask = fg_mask.bool()
        motif_tokens = self.fg_proj(fg) * fg_mask.unsqueeze(-1)
        motif_tokens, fg_mask = _align_token_and_mask(motif_tokens, fg_mask)

        protein_out = self.protein_encoder(protein)
        residue_tokens = protein_out["candidate"]
        residue_mask = protein_out["candidate_mask"]
        residue_tokens, residue_mask = _align_token_and_mask(residue_tokens, residue_mask)
        protein_global = protein_out["global"]
        pocket_tokens = protein_out["pocket"]
        pocket_mask = torch.ones(
            pocket_tokens.size(0),
            pocket_tokens.size(1),
            dtype=torch.bool,
            device=pocket_tokens.device
        )

        if self.fine_interaction_target == "residue":
            fine_target_tokens = protein_out["residue"]
            fine_target_mask = protein_out["residue_mask"]
        elif self.fine_interaction_target == "candidate":
            fine_target_tokens = residue_tokens
            fine_target_mask = residue_mask
        elif self.fine_interaction_target == "topk_residue":
            pocket_attention = protein_out["pocket_attention"]
            if pocket_attention.dim() == 4:
                residue_scores = pocket_attention.mean(dim=(1, 2))
            elif pocket_attention.dim() == 3:
                residue_scores = pocket_attention.mean(dim=1)
            else:
                residue_scores = protein_out["residue_mask"].float()
            fine_target_tokens, fine_target_mask, _, _ = _select_topk_tokens(
                protein_out["residue"],
                protein_out["residue_mask"],
                residue_scores,
                self.fine_residue_topk
            )
        else:
            fine_target_tokens = pocket_tokens
            fine_target_mask = pocket_mask

        evidence_out = self.evidence(motif_tokens, residue_tokens, fg_mask, residue_mask)
        local_logits = evidence_out["local_logits"]
        performance_out = self.performance_head(
            drug_tokens,
            drug_mask,
            motif_tokens,
            fg_mask,
            pocket_tokens,
            pocket_mask,
            fine_target_tokens,
            fine_target_mask,
            drug_global,
            protein_global
        )
        performance_logits = performance_out["logits"]

        global_feature = torch.cat([drug_global, protein_global], dim=-1)
        global_logits = self.global_calibrator(global_feature)
        global_gate = torch.sigmoid(self.global_gate(global_feature)) * self.global_calibration_max
        if self.prediction_mode == "performance_only":
            logits = performance_logits
        elif self.prediction_mode == "local_only":
            logits = local_logits
        elif self.prediction_mode == "global_only":
            logits = global_logits
        else:
            logits = performance_logits + self.evidence_aux_logit_weight * local_logits + global_gate * global_logits

        gate_sparsity_loss = global_gate.mean()
        global_calibration_penalty = (global_gate * global_logits).pow(2).mean()

        self.last_explanations = {
            "top_motif_indices": evidence_out["top_motif_indices"].detach(),
            "top_residue_indices": evidence_out["top_residue_indices"].detach(),
            "top_scores": evidence_out["top_scores"].detach(),
            "global_alpha": global_gate.detach(),
        }

        if return_aux:
            return {
                "logits": logits,
                "drug_logits": local_logits,
                "coarse_logits": global_logits,
                "aux": {
                    "evidence_sparsity_loss": evidence_out["evidence_sparsity_loss"],
                    "global_shortcut_loss": gate_sparsity_loss,
                    "global_calibration_penalty": global_calibration_penalty,
                    "global_alpha": global_gate,
                    "performance_feature_norm": performance_out["performance_feature_norm"],
                    "fine_feature_norm": performance_out["fine_feature_norm"],
                    "coarse_feature_norm": performance_out["coarse_feature_norm"],
                    "atom_pocket_gate_mean": performance_out["atom_pocket_gate_mean"],
                    "fine_target_count": performance_out["fine_target_count"],
                    "evidence_norm": evidence_out["evidence_norm"],
                    "pair_score_mean": evidence_out["pair_score_mean"],
                    "motif_attention_entropy": evidence_out["motif_attention_entropy"],
                    "residue_attention_entropy": evidence_out["residue_attention_entropy"],
                }
            }

        return logits, local_logits, global_logits
