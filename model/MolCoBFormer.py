# -*- coding: utf-8 -*-
# @Time     : 2025/4/19 上午10:39
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : MolCoBFormer.py
# @Project  : Graph_DTI
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import act_class_mapping, GCN, FAlignment, MultiHeadAttention
from .MolCoBlayer import BGA


def _make_sequence_mask(lengths, max_len, device):
    lengths = lengths.long().view(-1).clamp(min=0, max=max_len)
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


class MolCoBFormer(nn.Module):
    def __init__(self, config):
        super(MolCoBFormer, self).__init__()
        self.activation = act_class_mapping[config['activation'].lower()]()
        self.num_layers = config['num_layers']
        self.n_head = config['n_head']
        self.emb_dim = config['input_dim']
        self.out_dim = config['out_dim']
        self.num_hidden = config['num_hidden']

        self.fg_dim = config['fg_dim']
        self.layer_norm = config['layer_norm']
        self.batch_norm = config['batch_norm']
        self.residual = config['residual']

        self.atom_dropout = config['drug_dropout']
        self.graph_dropout = config['fc_dropout']

        self.fg_emb = nn.Linear(self.fg_dim, self.emb_dim * 2)
        self.fg_layers = nn.ModuleList([
            FGTransformer(self.emb_dim * 2, self.num_hidden * 2, self.emb_dim * 2, self.n_head,
                          self.graph_dropout, self.layer_norm, self.batch_norm, self.residual)
            for _ in range(self.num_layers)
        ])

        self.fgEmb = nn.Linear(self.emb_dim * 2, self.out_dim)
        self.fg_act = act_class_mapping[config['activation'].lower()]()

        num_atom_type = 108
        num_in_degree = config['num_in_degree']
        num_out_degree = config['num_out_degree']
        num_spatial_pos = config['num_spatial_pos']
        self.num_bond_type = config['bond_dim']

        self.embedding_atom = nn.Embedding(num_atom_type, self.emb_dim)
        self.embedding_in_degree = nn.Embedding(num_in_degree, self.emb_dim)
        self.embedding_out_degree = nn.Embedding(num_out_degree, self.emb_dim)
        self.embedding_spatial_pos = nn.Embedding(num_spatial_pos, self.emb_dim)
        self.embedding_e = nn.Embedding(self.num_bond_type, self.emb_dim)
        self.e_dis_encoder = nn.Embedding(3 * self.emb_dim * self.emb_dim, 1)
        self.in_feat_dropout = nn.Dropout(self.atom_dropout)

        self.gcn = GCN(self.emb_dim, self.num_hidden, self.emb_dim)

        self.gp_layers = nn.ModuleList([
            BGA(self.emb_dim, self.num_hidden, self.emb_dim, self.n_head,
                self.atom_dropout, self.layer_norm, self.batch_norm, self.residual)
            for _ in range(self.num_layers)
        ])

        self.aligner = FAlignment(self.emb_dim, self.emb_dim, self.n_head, temperature=0.1)

        self.topo_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.mp_layer = nn.Linear(self.emb_dim, self.out_dim)

        self.motif_retention_head = nn.Linear(self.out_dim, self.fg_dim)
        self.local_global_gate = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, 1)
        )
        self.last_aux = {}

    @staticmethod
    def _token_diversity_loss(tokens, mask=None, eps=1e-8):
        if tokens.numel() == 0 or tokens.size(1) <= 1:
            return tokens.sum() * 0.0
        tokens = F.normalize(tokens.float(), p=2, dim=-1, eps=eps)
        sim = torch.bmm(tokens, tokens.transpose(1, 2)).pow(2)
        eye = torch.eye(tokens.size(1), dtype=torch.bool, device=tokens.device).unsqueeze(0)
        valid = ~eye
        if mask is not None:
            mask = mask.bool()
            valid = valid & mask.unsqueeze(1) & mask.unsqueeze(2)
        if valid.sum() == 0:
            return tokens.sum() * 0.0
        return sim.masked_select(valid).mean()

    @staticmethod
    def _pad_atom_tokens(tokens, node_num, max_len=150):
        chunks = tokens.split(list(node_num), dim=0)
        padded = []
        for idx, chunk in enumerate(chunks):
            atom_count = int(node_num[idx].long().item())
            pad_len = max_len - atom_count
            if pad_len < 0:
                chunk = chunk[:max_len]
                pad_len = 0
            pad_emb = tokens.new_zeros((pad_len, tokens.size(-1)))
            padded.append(torch.cat([chunk, pad_emb], dim=0))
        return torch.stack(padded, dim=0)

    @staticmethod
    def _segment_mean_pool(tokens, node_num):
        pooled = []
        start = 0
        for atom_count in node_num:
            count = int(atom_count.long().item())
            end = start + count
            if count > 0:
                pooled.append(tokens[start:end].mean(dim=0))
            else:
                pooled.append(tokens.new_zeros(tokens.size(-1)))
            start = end
        return torch.stack(pooled, dim=0)

    def forward(self, g, e, node_num, fg, fg_num, in_degree, out_degree, spatial_pos, patch, patch_mask,
                return_aux=False):
        atom = self.embedding_atom(g.ndata['feat'])
        atom = atom + self.embedding_in_degree(in_degree) + self.embedding_out_degree(out_degree)
        atom = self.in_feat_dropout(atom)

        fg_emb = self.fg_emb(fg)
        fg_mask = fg_num.unsqueeze(1).float()
        fg_mask = torch.matmul(fg_mask.transpose(2, 1), fg_mask)

        for f_layer in self.fg_layers:
            fg_emb = f_layer(fg_emb, fg_mask)

        fg_feats = self.fgEmb(self.fg_act(fg_emb))

        raw_spatial_pos = spatial_pos.float()
        e = self.embedding_e(e)
        e_flat = e.permute(1, 0, 2)
        e_flat = torch.bmm(e_flat, self.e_dis_encoder.weight.reshape(-1, self.emb_dim, self.emb_dim))
        e = e_flat.sum(0) / (raw_spatial_pos.unsqueeze(-1).clamp(min=1.0))
        spatial_pos = self.embedding_spatial_pos(spatial_pos)

        topo_h = self.gcn(g, atom)
        atom_local = topo_h

        patch_mask = torch.matmul(patch_mask, patch_mask.transpose(3, 2)).int()
        patch_mask = torch.where(patch_mask == 0, 1, 0)

        distance_ratios = []
        for layer in self.gp_layers:
            atom, e, layer_stats = layer(
                g, atom, e, spatial_pos, patch, patch_mask,
                spatial_distance=raw_spatial_pos,
                return_attention=True
            )
            distance_ratios.append(layer_stats["attention_distance_ratio"])

        atom_global = atom
        local_summary = self._segment_mean_pool(atom_local, node_num)
        global_summary = self._segment_mean_pool(atom_global, node_num)
        gate = torch.sigmoid(self.local_global_gate(torch.cat([local_summary, global_summary], dim=-1)))
        expanded_gate = torch.repeat_interleave(gate, node_num.long().view(-1), dim=0)
        atom = (1.0 - expanded_gate) * atom_local + expanded_gate * atom

        batch_size = atom.shape[0]
        mol_mask = torch.eye(batch_size, dtype=torch.bool, device=atom.device)

        topo_aligned, mp_aligned, fused_feats = self.aligner(atom, topo_h)
        atom_align_loss = self.aligner.contrastive_loss(topo_aligned, mp_aligned, mol_mask)

        topo_atom = self.topo_linear(fused_feats)
        topo_atom = self.activation(topo_atom)
        topo_atom = self.mp_layer(topo_atom)

        y1 = self._pad_atom_tokens(topo_atom, node_num)
        token_mask = _make_sequence_mask(node_num, y1.size(1), y1.device)

        fg_recon = self.motif_retention_head(fg_feats)
        fg_valid = fg_num.view(fg_num.size(0), -1) > 0
        if fg_valid.any():
            motif_retention_loss = F.mse_loss(fg_recon[fg_valid], fg[fg_valid])
        else:
            motif_retention_loss = fg_recon.sum() * 0.0
        token_diversity = self._token_diversity_loss(y1, token_mask)
        local_global_penalty = gate.pow(2).mean()
        attention_distance_penalty = (
            torch.stack([v.to(y1.device).float() for v in distance_ratios]).mean()
            if distance_ratios else y1.sum() * 0.0
        )
        aux_loss = motif_retention_loss + token_diversity + local_global_penalty + attention_distance_penalty
        self.last_aux = {
            "align_loss": atom_align_loss,
            "aux_loss": aux_loss,
            "mean_local_global_gate": gate.detach().mean(),
        }

        if return_aux:
            return {
                "atom_tokens": y1,
                "motif_tokens": fg_feats,
                "align_loss": atom_align_loss,
                "aux": self.last_aux,
            }

        return [y1, fg_feats, atom_align_loss]


class FGTransformer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True,
                 residual=True):
        super(FGTransformer, self).__init__()
        self.in_channels = in_dim
        self.hid_channels = hid_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.act = nn.ReLU()

        self.att = MultiHeadAttention(num_heads, in_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(in_dim)
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(in_dim)
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

        self.FFN_h_layer1 = nn.Linear(in_dim, hid_dim)
        self.FFN_h_layer2 = nn.Linear(hid_dim, out_dim)

    def forward(self, fg, mask):
        h, _ = self.att(fg, fg, fg, mask)
        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h.permute(0, 2, 1))
            h = h.permute(0, 2, 1)

        h_in = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h.permute(0, 2, 1))
            h = h.permute(0, 2, 1)

        return h
