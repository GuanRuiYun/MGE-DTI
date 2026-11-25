# -*- coding: utf-8 -*-
# @Time     : 2025/4/18 下午10:53
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : utils.py
# @Project  : Graph_DTI
# @Description :
import dgl
import torch
import numpy as np
import torch.nn.functional as F
from itertools import chain
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    precision_recall_curve, auc, f1_score, matthews_corrcoef
)


def drug_collate(samples):
    graphs, nums, edge_input, fg_list, num_fg, \
        in_degree, out_degree, spatial_pos, drug_id, patch = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    nums = torch.tensor(np.array(nums)).unsqueeze(1)
    batch_size = len(nums)

    in_degree = torch.tensor(np.array(list(chain.from_iterable(in_degree))))
    out_degree = torch.tensor(np.array(list(chain.from_iterable(out_degree))))
    spatial_pos = torch.cat(list(spatial_pos))
    edge_input = torch.cat(edge_input, dim=0)

    fg_num = [sub[0] for sub in num_fg]
    max_fg_num = 14
    fg_dim = len(fg_list[0][0])

    fg_padded_list = []
    for sub_list in fg_list:
        sub_num = sub_list.shape[0]
        if sub_num < max_fg_num:
            temp = torch.cat((sub_list, torch.stack([
                torch.zeros(fg_dim) for _ in range(max_fg_num - sub_num)])), dim=0)
            fg_padded_list.append(temp)
        else:
            fg_padded_list.append(sub_list)

    fg_list = torch.stack(fg_padded_list)
    fg_num = torch.tensor(np.array(fg_num)).unsqueeze(1)
    fg_mask = (torch.arange(max_fg_num).expand(len(fg_num), max_fg_num) < fg_num)
    patch = torch.stack(patch)
    cumulative_nodes = np.cumsum(nums)
    offsets = np.concatenate([[0], cumulative_nodes[:-1]])
    offsets = offsets.reshape(len(nums), 1, 1)
    patch_matrix = patch + offsets

    target_values = (nums - 1).view(batch_size, 1, 1)
    patch_mask = (patch == target_values)
    patch_mask_matrix = patch_mask.float().unsqueeze(-1)

    return batched_graph, nums, edge_input, fg_list, fg_mask, \
        in_degree, out_degree, spatial_pos, drug_id, patch_matrix, patch_mask_matrix


def assert_matching_drug_ids(protein_drug_ids, drug_ids, split, batch_idx):
    if len(protein_drug_ids) != len(drug_ids):
        raise ValueError(
            "{} batch {} size mismatch: protein batch has {} drugs, molecule batch has {} drugs".format(
                split, batch_idx, len(protein_drug_ids), len(drug_ids)
            )
        )

    protein_drug_ids = [str(x) for x in protein_drug_ids]
    drug_ids = [str(x) for x in drug_ids]
    if protein_drug_ids != drug_ids:
        mismatches = [
            (i, protein_id, drug_id)
            for i, (protein_id, drug_id) in enumerate(zip(protein_drug_ids, drug_ids))
            if protein_id != drug_id
        ]
        preview = mismatches[:5]
        raise ValueError(
            "{} batch {} drug/protein dataloader mismatch. First mismatches: {}".format(
                split, batch_idx, preview
            )
        )


def extract_drug_ids_from_protein_pair(pair):
    return str(pair.strip().split()[-4])


def extract_drug_ids_from_protein_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise ValueError(
            "Protein DataLoader must return drug ids. Use create_collate_fn(return_drug_ids=True)."
        )
    return batch[2]


def extract_drug_ids_from_drug_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 9:
        raise ValueError("Unexpected molecule batch format; cannot locate drug ids.")
    return batch[8]


def validate_split_drug_ids(protein_dataset, drug_dataset, split):
    if not hasattr(protein_dataset, 'pairs') or not hasattr(drug_dataset, 'drug_id'):
        raise ValueError("{} split does not expose pairs/drug_id for preflight validation.".format(split))

    protein_ids = [extract_drug_ids_from_protein_pair(pair) for pair in protein_dataset.pairs]
    drug_ids = [str(drug_id) for drug_id in drug_dataset.drug_id]
    assert_matching_drug_ids(protein_ids, drug_ids, split, "dataset")


def validate_paired_dataloaders(protein_loader, drug_loader, split):
    if len(protein_loader.dataset) != len(drug_loader.dataset):
        raise ValueError(
            "{} dataset size mismatch: protein={} molecule={}".format(
                split, len(protein_loader.dataset), len(drug_loader.dataset)
            )
        )
    if len(protein_loader) != len(drug_loader):
        raise ValueError(
            "{} dataloader batch count mismatch: protein={} molecule={}".format(
                split, len(protein_loader), len(drug_loader)
            )
        )

    checked = 0
    for batch_idx, (protein_batch, drug_batch) in enumerate(zip(protein_loader, drug_loader)):
        protein_drug_ids = extract_drug_ids_from_protein_batch(protein_batch)
        drug_ids = extract_drug_ids_from_drug_batch(drug_batch)
        assert_matching_drug_ids(protein_drug_ids, drug_ids, split, batch_idx)
        checked += len(protein_drug_ids)

    print("{} dataloader alignment verified: {} samples, {} batches.".format(
        split, checked, len(protein_loader)
    ))


def prc_auc_score(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def binary_metrics_from_scores(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= float(threshold)).astype(int)
    if len(np.unique(labels)) < 2:
        auc_value = 0.5
        prc_value = 0.0
    else:
        auc_value = roc_auc_score(labels, scores)
        prc_value = prc_auc_score(labels, scores)
    return {
        "threshold": float(threshold),
        "predictions": predictions,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "auc": auc_value,
        "prc": prc_value,
        "f1": f1_score(labels, predictions, average='binary', zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "sensitivity": recall_score(labels, predictions, zero_division=0),
        "specificity": recall_score(labels, predictions, pos_label=0, zero_division=0),
    }


def _score_curve_metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0
    return roc_auc_score(labels, scores), prc_auc_score(labels, scores)


def _binary_metrics_from_predictions(labels, predictions, threshold, auc_value, prc_value):
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    recall = recall_score(labels, predictions, zero_division=0)
    specificity = recall_score(labels, predictions, pos_label=0, zero_division=0)
    return {
        "threshold": float(threshold),
        "predictions": predictions,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall,
        "auc": float(auc_value),
        "prc": float(prc_value),
        "f1": f1_score(labels, predictions, average='binary', zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "sensitivity": recall,
        "specificity": specificity,
    }


def _threshold_primary_score(metrics, metric):
    if metric == "f1_mcc":
        return 0.5 * float(metrics["f1"]) + 0.5 * ((float(metrics["mcc"]) + 1.0) / 2.0)
    if metric == "youden":
        return float(metrics["sensitivity"]) + float(metrics["specificity"]) - 1.0
    return float(metrics.get(metric, metrics["mcc"]))


def find_best_threshold(labels, scores, metric="f1_mcc", auc_value=None, prc_value=None):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    grid = np.linspace(0.1, 0.9, 33, dtype=float)
    if auc_value is None or prc_value is None:
        auc_value, prc_value = _score_curve_metrics(labels, scores)
    best_threshold = 0.5
    best_predictions = (scores >= best_threshold).astype(int)
    best_metrics = _binary_metrics_from_predictions(
        labels, best_predictions, best_threshold, auc_value, prc_value
    )
    best_key = (
        _threshold_primary_score(best_metrics, metric),
        best_metrics["mcc"],
        best_metrics["f1"],
        best_metrics["accuracy"],
        -abs(best_threshold - 0.5),
    )
    for threshold in grid:
        predictions = (scores >= float(threshold)).astype(int)
        metrics = _binary_metrics_from_predictions(
            labels, predictions, threshold, auc_value, prc_value
        )
        key = (
            _threshold_primary_score(metrics, metric),
            metrics["mcc"],
            metrics["f1"],
            metrics["accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def validation_selection_score(auc_value, prc_value, mcc_value, f1_value, acc_value, args):
    if getattr(args, "selection_metric", "balanced") == "auc_acc":
        return auc_acc_score(auc_value, acc_value, args.auc_weight)

    weights = np.asarray([
        float(args.auc_weight),
        float(getattr(args, "aupr_weight", 0.0)),
        float(getattr(args, "mcc_weight", 0.0)),
        float(getattr(args, "f1_weight", 0.0)),
    ], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("At least one balanced selection weight must be positive.")
    weights = weights / weights.sum()
    mcc_norm = (float(mcc_value) + 1.0) / 2.0
    values = np.asarray([auc_value, prc_value, mcc_norm, f1_value], dtype=float)
    return float(np.dot(weights, values))


def auc_acc_score(auc_value, acc_value, auc_weight=0.5):
    auc_weight = float(auc_weight)
    if not 0.0 <= auc_weight <= 1.0:
        raise ValueError("auc_weight must be in [0, 1], got {}".format(auc_weight))
    return auc_weight * float(auc_value) + (1.0 - auc_weight) * float(acc_value)


def is_better_auc_acc(score, auc_value, acc_value, best_score, best_auc, best_acc, eps=1e-12):
    if score > best_score + eps:
        return True
    if abs(score - best_score) <= eps and auc_value > best_auc + eps:
        return True
    if abs(score - best_score) <= eps and abs(auc_value - best_auc) <= eps and acc_value > best_acc + eps:
        return True
    return False


def split_weight_bias_params(module, include_frozen=False):
    weight_params, bias_params = [], []
    for name, param in module.named_parameters():
        if not include_frozen and not param.requires_grad:
            continue
        if 'bias' in name or param.dim() <= 1:
            bias_params.append(param)
        else:
            weight_params.append(param)
    return weight_params, bias_params


def weighted_loss_term(loss, weight):
    weight = float(weight)
    if weight == 0.0:
        return loss.detach() * 0.0
    return loss * weight


def scalar_metric_value(value, default=0.0):
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def module_has_trainable_params(module):
    return any(param.requires_grad for param in module.parameters())


def mol_unfreeze_strategy(args):
    if getattr(args, "unfreeze_molcobformer", False):
        return "full"
    return getattr(args, "mol_unfreeze_strategy", "none")


def set_molcobformer_trainable(model_drug, args, epoch=None):
    strategy = mol_unfreeze_strategy(args)
    if strategy == "none":
        model_drug.requires_grad_(False)
        model_drug.eval()
        return False, "frozen"

    if strategy == "full":
        model_drug.requires_grad_(True)
        model_drug.train()
        return True, "full"

    if strategy != "staged":
        raise ValueError("Unknown mol_unfreeze_strategy: {}".format(strategy))

    if epoch is None:
        epoch = 1
    stage1 = int(getattr(args, "mol_stage1_epochs", 5))
    stage2 = int(getattr(args, "mol_stage2_epochs", 15))

    model_drug.requires_grad_(False)
    if epoch <= stage1:
        model_drug.eval()
        return False, "stage1_frozen"

    trainable_names = (
        "fg_emb", "fg_layers", "fgEmb", "motif_retention_head",
        "topo_linear", "mp_layer", "local_global_gate"
    )
    for name, param in model_drug.named_parameters():
        if any(key in name for key in trainable_names):
            param.requires_grad = True

    if hasattr(model_drug, "gp_layers") and len(model_drug.gp_layers) > 0:
        for param in model_drug.gp_layers[-1].parameters():
            param.requires_grad = True

    if epoch > stage2:
        if hasattr(model_drug, "gp_layers"):
            for layer in model_drug.gp_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        if hasattr(model_drug, "gcn"):
            for param in model_drug.gcn.parameters():
                param.requires_grad = True
        stage_name = "stage3_controlled_bga"
    else:
        stage_name = "stage2_fg_local_last_bga"

    model_drug.train()
    return module_has_trainable_params(model_drug), stage_name


def pairwise_auc_hard_negative_loss(scores, labels, margin=1.0, hard_topk_frac=1.0):
    scores = scores.float().view(-1)
    labels = labels.long().view(-1)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return scores.sum() * 0.0

    losses = F.softplus(float(margin) - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))).reshape(-1)
    hard_topk_frac = float(hard_topk_frac)
    if hard_topk_frac >= 1.0:
        return losses.mean()
    if hard_topk_frac <= 0.0:
        raise ValueError("ranking_hard_topk_frac must be in (0, 1], got {}".format(hard_topk_frac))
    keep = max(1, int(np.ceil(losses.numel() * hard_topk_frac)))
    hard_losses, _ = torch.topk(losses, k=keep, largest=True, sorted=False)
    return hard_losses.mean()


def unpack_drug_encoder_outputs(outputs):
    if isinstance(outputs, dict):
        atom_tokens = outputs["atom_tokens"]
        motif_tokens = outputs["motif_tokens"]
        aux = outputs.get("aux", {})
        align_loss = outputs.get("align_loss", aux.get("align_loss", atom_tokens.sum() * 0.0))
        return atom_tokens, motif_tokens, align_loss, aux

    atom_tokens, motif_tokens, align_loss = outputs
    return atom_tokens, motif_tokens, align_loss, {}


def unpack_model_outputs(outputs):
    if isinstance(outputs, dict):
        logits = outputs["logits"]
        drug_logits = outputs.get("drug_logits", logits)
        coarse_logits = outputs.get("coarse_logits", logits)
        aux = outputs.get("aux", {})
        return logits, drug_logits, coarse_logits, aux

    logits, drug_logits, coarse_logits = outputs
    return logits, drug_logits, coarse_logits, {}


def apply_temperature_to_scores(scores, temperature):
    scores = np.asarray(scores, dtype=float)
    temperature = max(float(temperature), 1e-6)
    scores = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(scores / (1.0 - scores))
    calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def find_best_temperature_and_threshold(labels, scores, args):
    temp_min = float(getattr(args, "temperature_min", 0.7))
    temp_max = float(getattr(args, "temperature_max", 1.6))
    grid_size = int(getattr(args, "temperature_grid_size", 19))
    if grid_size <= 1:
        temperature_grid = np.asarray([1.0], dtype=float)
    else:
        temperature_grid = np.linspace(temp_min, temp_max, grid_size, dtype=float)
    best = None
    for temperature in temperature_grid:
        calibrated_scores = apply_temperature_to_scores(scores, temperature)
        auc_value, prc_value = _score_curve_metrics(labels, calibrated_scores)
        threshold, metrics = find_best_threshold(
            labels,
            calibrated_scores,
            metric=getattr(args, "threshold_metric", "f1_mcc"),
            auc_value=auc_value,
            prc_value=prc_value,
        )
        key = (
            _threshold_primary_score(metrics, getattr(args, "threshold_metric", "f1_mcc")),
            metrics["mcc"],
            metrics["f1"],
            metrics["accuracy"],
            -abs(float(temperature) - 1.0),
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best["key"]:
            best = {
                "temperature": float(temperature),
                "threshold": float(threshold),
                "metrics": metrics,
                "scores": calibrated_scores,
                "key": key,
            }
    return best
