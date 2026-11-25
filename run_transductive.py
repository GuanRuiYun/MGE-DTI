# -*- coding: utf-8 -*-
# @Time     : 2025/12/7 下午4:32
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : run_transductive.py
# @Project  : Graph_DTI
# @Description :


import os
import copy
import csv
import yaml
import torch
import random
import timeit
import numpy as np
import os.path as osp
from yaml import SafeLoader
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from parsing import parse_args
from pytorchtools import EarlyStopping
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from utils import (
    apply_temperature_to_scores,
    assert_matching_drug_ids,
    binary_metrics_from_scores,
    drug_collate,
    find_best_temperature_and_threshold,
    find_best_threshold,
    is_better_auc_acc,
    mol_unfreeze_strategy,
    pairwise_auc_hard_negative_loss,
    scalar_metric_value,
    set_molcobformer_trainable,
    split_weight_bias_params,
    unpack_drug_encoder_outputs,
    unpack_model_outputs,
    validate_paired_dataloaders,
    validate_split_drug_ids,
    validation_selection_score,
    weighted_loss_term,
)
from losses import binary_brier_loss
from datapreprocess.get_data import get_data
from datapreprocess.proteins import create_collate_fn
from model import MolCoBFormer, EvidenceDTI
# Required in __main__ so pickle.load can resolve pickled data/molecule_data/*.pkl objects.
from datapreprocess.run import MoleculeDatasetDGL, MoleculeDGL  # noqa: F401
import warnings
warnings.filterwarnings('ignore')


def build_protein_collate(args, data_path, return_drug_ids=False):
    return create_collate_fn(
        return_drug_ids=return_drug_ids,
        protein_input_mode=getattr(args, "protein_input_mode", "esm"),
        data_root=data_path,
        dataset=args.dataset,
        esm_model=getattr(args, "esm_model_name", "esm2_t33_650M_UR50D")
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setting_result_dir(dataset, setting_name):
    return "{}_{}_setting".format(dataset, setting_name)


def count_model_parameters(model):
    """Return (total_params, trainable_params, non_trainable_params)."""
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

    return total_params, trainable_params, non_trainable_params


def build_optimizer(model, model_drug, learning_rate, weight_decay, args):
    protein_weight_p, protein_bias_p = split_weight_bias_params(model)
    param_groups = [
        {'params': protein_weight_p, 'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': protein_bias_p, 'weight_decay': 0.0, 'lr': learning_rate},
    ]

    if mol_unfreeze_strategy(args) != "none":
        drug_weight_p, drug_bias_p = split_weight_bias_params(model_drug, include_frozen=True)
        drug_lr = learning_rate * args.mol_lr_mult
        param_groups.extend([
            {'params': drug_weight_p, 'weight_decay': weight_decay, 'lr': drug_lr},
            {'params': drug_bias_p, 'weight_decay': 0.0, 'lr': drug_lr},
        ])

    return optim.AdamW(param_groups)


def build_class_weights(args, train_dataset, device):
    strategy = getattr(args, "class_weight_strategy", "default")
    if strategy == "none":
        print("Class weights: disabled")
        return None

    if strategy == "custom":
        raw_values = str(getattr(args, "class_weight_values", "")).strip()
        values = [float(item) for item in raw_values.split(",") if item.strip()]
        if len(values) != 2:
            raise ValueError("--class_weight_values must contain exactly two comma-separated values.")
        weights = torch.FloatTensor(values).to(device)
        print("Class weights: custom {}".format(values))
        return weights

    if strategy == "balanced":
        labels = []
        for pair in getattr(train_dataset, "pairs", []):
            labels.append(int(str(pair).strip().split()[-1]))
        if not labels:
            raise ValueError("Cannot compute balanced class weights from an empty train split.")
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (2.0 * counts)
        weights = weights / weights.mean()
        print("Class weights: balanced counts={} weights={}".format(
            counts.astype(int).tolist(), weights.tolist()))
        return torch.FloatTensor(weights).to(device)

    if args.dataset == "Davis":
        weights = [0.3, 0.7]
    else:
        weights = [0.2, 0.8]
    print("Class weights: default {}".format(weights))
    return torch.FloatTensor(weights).to(device)


def show_result(dataset, lable, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List,
                F1_List, MCC_List, Sensitivity_List, Specificity_List, setting_name="transductive"):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    F1_mean, F1_var = np.mean(F1_List), np.var(F1_List)
    mcc_mean, mcc_var = np.mean(MCC_List), np.var(MCC_List)
    sen_mean, sen_var = np.mean(Sensitivity_List), np.var(Sensitivity_List)
    spc_mean, spc_var = np.mean(Specificity_List), np.var(Specificity_List)
    print("The {} model's results:".format(lable))
    cur_path = os.getcwd()
    save_path = os.path.join(cur_path, setting_result_dir(dataset, setting_name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "results.txt"), 'a') as f:
        f.write("The {} model's wfg interaction results:".format(lable) + '\n')
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
        f.write('F1(std):{:.4f}({:.4f})'.format(F1_mean, F1_var) + '\n')
        f.write('MCC(std):{:.4f}({:.4f})'.format(mcc_mean, mcc_var) + '\n')
        f.write('Sensitivity(std):{:.4f}({:.4f})'.format(sen_mean, sen_var) + '\n')
        f.write('Spec(std):{:.4f}({:.4f})'.format(spc_mean, spc_var) + '\n')
        f.write('  \n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))
    print('F1(std):{:.4f}({:.4f})'.format(F1_mean, F1_var))
    print('MCC(std):{:.4f}({:.4f})'.format(mcc_mean, mcc_var))
    print('Sensitivity(std):{:.4f}({:.4f})'.format(sen_mean, sen_var))
    print('Spec(std):{:.4f}({:.4f})'.format(spc_mean, spc_var))


def test_precess(model, model_drug, pbar, drug_load, LOSS, device, threshold=0.5, temperature=1.0):
    model.eval()
    model_drug.eval()
    test_losses = []
    Y, P, S = [], [], []
    dataloader_iterator = iter(drug_load)

    with torch.no_grad():
        for i, data in pbar:
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break

            batch_graphs, num_atoms, edge_inputs, fg, fg_mask, \
                in_degrees, out_degrees, spatial_pos, drug_idx, patch, patch_mask = batch

            batch_graphs = batch_graphs.to(device)
            num_atoms = num_atoms.to(device)
            in_degrees = in_degrees.to(device)
            out_degrees = out_degrees.to(device)
            edge_inputs = edge_inputs.to(device)
            spatial_pos = spatial_pos.to(device)
            fg = fg.to(device)
            fg_mask = fg_mask.to(device)
            patch = patch.to(device)
            patch_mask = patch_mask.to(device)

            pred = model_drug(
                batch_graphs, edge_inputs, num_atoms, fg, fg_mask,
                in_degrees, out_degrees, spatial_pos, patch, patch_mask,
                return_aux=True
            )

            p1, p2, _, _ = unpack_drug_encoder_outputs(pred)
            proteins, labels, protein_drug_ids = data
            assert_matching_drug_ids(protein_drug_ids, drug_idx, "test", i)
            proteins = proteins.to(device)
            labels = labels.to(device)

            outputs = model(p1, p2, num_atoms, proteins, fg_mask)
            main_logits, _, _, _ = unpack_model_outputs(outputs)

            loss = LOSS(main_logits, labels)
            correct_labels = labels.detach().cpu().numpy()
            pred_score = F.softmax(main_logits, 1).detach().cpu().numpy()[:, 1]
            pred_score = apply_temperature_to_scores(pred_score, temperature)
            predicted_labels = (pred_score >= float(threshold)).astype(int)

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(pred_score)
            test_losses.append(loss.item())
    metrics = binary_metrics_from_scores(Y, S, threshold)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, metrics["accuracy"], metrics["precision"], metrics["recall"], \
        metrics["auc"], metrics["prc"], metrics["f1"], metrics["mcc"], metrics["sensitivity"], \
        metrics["specificity"]


def test_model(model, drug_model, dataset_load, drug_load, save_path, DATASET, LOSS, device, dataset="Train",
               lable="best", save=True, threshold=0.5, temperature=1.0):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, \
        f1_test, mcc, sensitivity, specificity = test_precess(
            model, drug_model, test_pbar, drug_load, LOSS, device, threshold=threshold, temperature=temperature)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET, dataset, lable), 'w') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Threshold:{:.3f};Temperature:{:.3f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f},' \
              'F1 Score:{:.5f}, MCC: {:.5f}, sensitivity: {:.5f}, specificity: {:.5f}.' \
        .format(lable, loss_test, threshold, temperature, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test,
                f1_test, mcc, sensitivity, specificity)
    print(results)

    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, f1_test, mcc, sensitivity, specificity


def _split_pair_record(pair):
    fields = str(pair).strip().split()
    return fields[-4], fields[-3], fields[-2], int(fields[-1])


def export_test_evidence(model, model_drug, dataset_load, drug_load, output_path, device,
                         threshold=0.5, temperature=1.0, max_batches=0):
    if not output_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()
    model_drug.eval()
    rows = []
    dataloader_iterator = iter(drug_load)
    exported_batches = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_load):
            if max_batches and exported_batches >= int(max_batches):
                break
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break

            batch_graphs, num_atoms, edge_inputs, fg, fg_mask, \
                in_degrees, out_degrees, spatial_pos, drug_idx, patch, patch_mask = batch

            batch_graphs = batch_graphs.to(device)
            num_atoms = num_atoms.to(device)
            in_degrees = in_degrees.to(device)
            out_degrees = out_degrees.to(device)
            edge_inputs = edge_inputs.to(device)
            spatial_pos = spatial_pos.to(device)
            fg = fg.to(device)
            fg_mask = fg_mask.to(device)
            patch = patch.to(device)
            patch_mask = patch_mask.to(device)

            drug_outputs = model_drug(
                batch_graphs, edge_inputs, num_atoms, fg, fg_mask,
                in_degrees, out_degrees, spatial_pos, patch, patch_mask,
                return_aux=True
            )
            drug_feats, fg_feats, _, _ = unpack_drug_encoder_outputs(drug_outputs)
            proteins, labels, protein_drug_ids = data
            assert_matching_drug_ids(protein_drug_ids, drug_idx, "evidence_export", batch_idx)
            proteins = proteins.to(device)
            labels = labels.to(device)
            outputs = model(drug_feats, fg_feats, num_atoms, proteins, fg_mask, return_aux=True)
            logits, _, _, _ = unpack_model_outputs(outputs)
            probs = F.softmax(logits, 1).detach().cpu().numpy()[:, 1]
            probs = apply_temperature_to_scores(probs, temperature)
            preds = (probs >= float(threshold)).astype(int)
            explanations = getattr(model, "last_explanations", {})
            top_scores = explanations.get("top_scores")
            top_motifs = explanations.get("top_motif_indices")
            top_candidates = explanations.get("top_residue_indices")
            global_alpha = explanations.get("global_alpha")
            if top_scores is None or top_motifs is None or top_candidates is None:
                continue
            top_scores = top_scores.detach().cpu()
            top_motifs = top_motifs.detach().cpu()
            top_candidates = top_candidates.detach().cpu()
            global_alpha = global_alpha.detach().cpu().view(-1).numpy() if global_alpha is not None else None
            residue_lengths = None
            if isinstance(proteins, dict) and "esm_lengths" in proteins:
                residue_lengths = proteins["esm_lengths"].detach().cpu().numpy()
            elif isinstance(proteins, dict) and "raw_tokens" in proteins:
                residue_lengths = (proteins["raw_tokens"].detach().cpu().numpy() > 0).sum(axis=1)

            pairs = getattr(dataset_load.dataset, "pairs", None)
            batch_size = len(labels)
            for sample_idx in range(batch_size):
                pair_idx = batch_idx * dataset_load.batch_size + sample_idx
                if pairs is not None and pair_idx < len(pairs):
                    drug_id, protein_id, protein_seq, label = _split_pair_record(pairs[pair_idx])
                else:
                    drug_id = str(protein_drug_ids[sample_idx])
                    protein_id = ""
                    protein_seq = ""
                    label = int(labels.detach().cpu()[sample_idx].item())
                residue_len = int(residue_lengths[sample_idx]) if residue_lengths is not None else len(protein_seq)
                for rank_idx in range(top_scores.size(1)):
                    candidate_index = int(top_candidates[sample_idx, rank_idx].item())
                    source = "residue" if candidate_index < residue_len else "pocket_slot"
                    source_index = candidate_index if source == "residue" else candidate_index - residue_len
                    rows.append({
                        "batch": batch_idx,
                        "sample_in_batch": sample_idx,
                        "drug_id": drug_id,
                        "protein_id": protein_id,
                        "label": label,
                        "pred": int(preds[sample_idx]),
                        "positive_score": float(probs[sample_idx]),
                        "global_alpha": "" if global_alpha is None else float(global_alpha[sample_idx]),
                        "evidence_rank": rank_idx + 1,
                        "motif_index": int(top_motifs[sample_idx, rank_idx].item()),
                        "candidate_index": candidate_index,
                        "candidate_source": source,
                        "source_index": source_index,
                        "evidence_score": float(top_scores[sample_idx, rank_idx].item()),
                    })
            exported_batches += 1

    fieldnames = [
        "batch", "sample_in_batch", "drug_id", "protein_id", "label", "pred",
        "positive_score", "global_alpha", "evidence_rank", "motif_index",
        "candidate_index", "candidate_source", "source_index", "evidence_score"
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Exported {} evidence rows to {}".format(len(rows), output_path))


def run(args, config, device, drug_data, p_train_data, p_val_data, p_test_data, TVdata_len, setting_name=None):
    g = torch.Generator()
    g.manual_seed(config['seed'])
    num_works = args.num_works
    dp_batch = config['batch_size']
    data_path = osp.join(os.getcwd(), "data")
    protein_collate_with_ids = build_protein_collate(args, data_path, return_drug_ids=True)
    train_dataset_load = DataLoader(p_train_data, batch_size=dp_batch, shuffle=True, num_workers=num_works,
                                    collate_fn=protein_collate_with_ids, generator=g)
    valid_dataset_load = DataLoader(p_val_data, batch_size=dp_batch, shuffle=False, num_workers=num_works,
                                    collate_fn=protein_collate_with_ids)
    test_dataset_load = DataLoader(p_test_data, batch_size=dp_batch, shuffle=False, num_workers=num_works,
                                   collate_fn=protein_collate_with_ids)

    g = torch.Generator()
    g.manual_seed(config['seed'])
    train_drug_dataset, valid_drug_dataset, test_drug_dataset = drug_data.train, drug_data.val, drug_data.test
    train_drug_load = DataLoader(train_drug_dataset, batch_size=dp_batch, shuffle=True, num_workers=num_works,
                                 collate_fn=drug_collate, generator=g)
    valid_drug_load = DataLoader(valid_drug_dataset, batch_size=dp_batch, shuffle=False, num_workers=num_works,
                                 collate_fn=drug_collate)
    test_drug_load = DataLoader(test_drug_dataset, batch_size=dp_batch, shuffle=False, num_workers=num_works,
                                collate_fn=drug_collate)

    validate_split_drug_ids(p_train_data, train_drug_dataset, "train")
    validate_split_drug_ids(p_val_data, valid_drug_dataset, "valid")
    validate_split_drug_ids(p_test_data, test_drug_dataset, "test")
    validate_paired_dataloaders(train_dataset_load, train_drug_load, "train")
    validate_paired_dataloaders(valid_dataset_load, valid_drug_load, "valid")
    validate_paired_dataloaders(test_dataset_load, test_drug_load, "test")

    g = torch.Generator()
    g.manual_seed(config['seed'])
    train_dataset_load = DataLoader(p_train_data, batch_size=dp_batch, shuffle=True, num_workers=num_works,
                                    collate_fn=protein_collate_with_ids, generator=g)
    g = torch.Generator()
    g.manual_seed(config['seed'])
    train_drug_load = DataLoader(train_drug_dataset, batch_size=dp_batch, shuffle=True, num_workers=num_works,
                                 collate_fn=drug_collate, generator=g)

    learning_rate = config['learning_rate']
    weight_decay = args.weight_decay
    num_epochs = config['num_epochs']
    train_type = args.train_type
    dataset = args.dataset
    if setting_name is None:
        setting_name = train_type

    model = EvidenceDTI(args, config, device).to(device)
    model_drug = MolCoBFormer(config).to(device)
    set_molcobformer_trainable(model_drug, args, epoch=0)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model_parameters = count_model_parameters(model)
    model_drug_parameters = count_model_parameters(model_drug)

    print("Model: " + str(model_parameters))
    print("Drug model: " + str(model_drug_parameters))

    weight_CE = build_class_weights(args, p_train_data, device)

    optimizer = build_optimizer(model, model_drug, learning_rate, weight_decay, args)

    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=[group['lr'] for group in optimizer.param_groups],
        max_lr=[group['lr'] * 5 for group in optimizer.param_groups],
        cycle_momentum=False,
        step_size_up=max(1, int(((1 - 0.2) * TVdata_len) // dp_batch)))

    Loss = nn.CrossEntropyLoss(weight=weight_CE)
    Loss_D = nn.CrossEntropyLoss(weight=weight_CE)
    Loss_F = nn.CrossEntropyLoss(weight=weight_CE)
    l1 = args.main_loss_weight
    l2 = args.local_loss_weight
    l_global = args.global_loss_weight
    l_auc = args.pairwise_auc_weight
    l_mol_aux = args.mol_aux_loss_weight
    l_mol_align = args.mol_align_loss_weight
    l_sparse = args.evidence_sparsity_weight
    l_shortcut = args.shortcut_loss_weight
    l_calib = args.calibration_loss_weight

    mol_mode = mol_unfreeze_strategy(args)
    if args.selection_metric == "balanced":
        selection_desc = (
            "balanced={:.2f}*AUC+{:.2f}*AUPR+{:.2f}*MCC_norm+{:.2f}*F1"
        ).format(args.auc_weight, args.aupr_weight, args.mcc_weight, args.f1_weight)
    else:
        selection_desc = "auc_acc={:.2f}*AUC+{:.2f}*ACC".format(
            args.auc_weight, 1.0 - args.auc_weight)
    loss_desc = (
        "loss=main*{:.2f}+local*{:.2f}+global*{:.2f}+auc*{:.2f}; "
        "optional aux weights: mol_aux={:.2f}, align={:.2f}, sparse={:.2f}, shortcut={:.2f}, calib={:.2f}; "
        "EvidenceDTI is an auxiliary explanation head"
    ).format(l1, l2, l_global, l_auc, l_mol_aux, l_mol_align, l_sparse, l_shortcut, l_calib)
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0)) or int(config['patience'])
    train_desc = (
        "EvidenceDTI; scoring=coarse_to_fine_atom_pocket_with_aux_evidence; "
        "MolCoBFormer={}; selection={}; patience={}; {}"
    ).format(mol_mode, selection_desc, early_stopping_patience, loss_desc)
    print("Training " + train_desc)

    result_dir = os.path.join(os.getcwd(), setting_result_dir(dataset, setting_name))
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "results.txt"), 'a') as f:
        f.write(train_desc + "\n")

    print('Training...')
    cur_path = os.getcwd()
    save_path = os.path.join(cur_path, "molecule_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, delta=0)

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    F1_Score_List_stable, MCC_List_stable, Sen_List_Stable, Spc_List_Stable = [], [], [], []
    start_time = timeit.default_timer()

    best_acc = 0.0
    best_auc = 0.0
    best_score = -float('inf')
    best_threshold = 0.5
    best_temperature = 1.0
    best_threshold_metrics = None
    latest_calibration_result = None
    calibration_interval = max(1, int(getattr(args, "temperature_update_interval", 5)))

    best_drug_model = None
    best_protein_model = None

    for epoch in range(1, num_epochs + 1):
        trian_pbar = tqdm(
            enumerate(
                BackgroundGenerator(train_dataset_load)),
            total=len(train_dataset_load))

        train_losses_in_epoch = []
        train_main_losses_in_epoch = []
        train_local_losses_in_epoch = []
        train_global_losses_in_epoch = []
        train_auc_losses_in_epoch = []
        train_mol_aux_losses_in_epoch = []
        train_sparse_losses_in_epoch = []
        train_shortcut_losses_in_epoch = []
        train_calib_losses_in_epoch = []
        train_gate_means_in_epoch = []
        train_evidence_norms_in_epoch = []
        train_pair_score_means_in_epoch = []
        train_motif_entropy_in_epoch = []
        train_residue_entropy_in_epoch = []
        train_global_alpha_in_epoch = []
        train_perf_norms_in_epoch = []
        train_fine_norms_in_epoch = []
        train_coarse_norms_in_epoch = []
        train_atom_gate_in_epoch = []
        train_fine_token_counts_in_epoch = []

        model.train()
        mol_trainable, mol_stage = set_molcobformer_trainable(model_drug, args, epoch=epoch)

        dataloader_iterator = iter(train_drug_load)
        for trian_i, train_data in trian_pbar:
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break

            batch_graphs, num_atoms, edge_inputs, fg, fg_mask, \
                in_degrees, out_degrees, spatial_pos, drug_idx, patch, patch_mask = batch

            batch_graphs = batch_graphs.to(device)
            num_atoms = num_atoms.to(device)
            in_degrees = in_degrees.to(device)
            out_degrees = out_degrees.to(device)
            edge_inputs = edge_inputs.to(device)
            spatial_pos = spatial_pos.to(device)
            fg = fg.to(device)
            fg_mask = fg_mask.to(device)
            patch = patch.to(device)
            patch_mask = patch_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            if mol_trainable:
                drug_outputs = model_drug(
                    batch_graphs, edge_inputs, num_atoms, fg, fg_mask,
                    in_degrees, out_degrees, spatial_pos, patch, patch_mask,
                    return_aux=True
                )
            else:
                with torch.no_grad():
                    drug_outputs = model_drug(
                        batch_graphs, edge_inputs, num_atoms, fg, fg_mask,
                        in_degrees, out_degrees, spatial_pos, patch, patch_mask,
                        return_aux=True
                    )
            drug_feats, fg_feats, d_loss, drug_aux = unpack_drug_encoder_outputs(drug_outputs)

            train_proteins, train_labels, train_drug_ids = train_data
            assert_matching_drug_ids(train_drug_ids, drug_idx, "train", trian_i)
            train_proteins = train_proteins.to(device)
            train_labels = train_labels.to(device)

            outputs = model(drug_feats, fg_feats, num_atoms, train_proteins, fg_mask, return_aux=True)
            predicted, d_predict, f_predict, model_aux = unpack_model_outputs(outputs)

            train_loss = Loss(predicted, train_labels)
            drug_train_loss = Loss_D(d_predict, train_labels)
            fg_train_loss = Loss_F(f_predict, train_labels)
            auc_loss = pairwise_auc_hard_negative_loss(
                F.softmax(predicted, 1)[:, 1],
                train_labels,
                margin=args.ranking_margin,
                hard_topk_frac=args.ranking_hard_topk_frac
            )
            align_loss = drug_aux.get("align_loss", d_loss)
            mol_aux_loss = drug_aux.get("aux_loss", getattr(model_drug, "last_aux", {}).get("aux_loss", d_loss))
            sparse_loss = model_aux.get("evidence_sparsity_loss", predicted.sum() * 0.0)
            shortcut_loss = model_aux.get("global_shortcut_loss", predicted.sum() * 0.0)
            calibration_loss = binary_brier_loss(predicted, train_labels)
            mean_local_global_gate = drug_aux.get("mean_local_global_gate", predicted.sum() * 0.0)
            evidence_norm = model_aux.get("evidence_norm", predicted.sum() * 0.0)
            pair_score_mean = model_aux.get("pair_score_mean", predicted.sum() * 0.0)
            motif_attention_entropy = model_aux.get("motif_attention_entropy", predicted.sum() * 0.0)
            residue_attention_entropy = model_aux.get("residue_attention_entropy", predicted.sum() * 0.0)
            global_alpha = model_aux.get("global_alpha", predicted.sum() * 0.0)
            performance_feature_norm = model_aux.get("performance_feature_norm", predicted.sum() * 0.0)
            fine_feature_norm = model_aux.get("fine_feature_norm", predicted.sum() * 0.0)
            coarse_feature_norm = model_aux.get("coarse_feature_norm", predicted.sum() * 0.0)
            atom_pocket_gate_mean = model_aux.get("atom_pocket_gate_mean", predicted.sum() * 0.0)
            fine_target_count = model_aux.get("fine_target_count", predicted.sum() * 0.0)

            loss = (
                weighted_loss_term(train_loss, l1)
                + weighted_loss_term(drug_train_loss, l2)
                + weighted_loss_term(fg_train_loss, l_global)
                + weighted_loss_term(auc_loss, l_auc)
                + weighted_loss_term(sparse_loss, l_sparse)
                + weighted_loss_term(shortcut_loss, l_shortcut)
                + weighted_loss_term(
                    calibration_loss + model_aux.get("global_calibration_penalty", predicted.sum() * 0.0),
                    l_calib
                )
                + weighted_loss_term(align_loss, l_mol_align)
                + weighted_loss_term(mol_aux_loss, l_mol_aux)
            )
            if not torch.isfinite(loss):
                print(
                    "Skip non-finite batch {}: train_loss={}, drug_loss={}, fg_loss={}, auc_loss={}".format(
                        trian_i,
                        train_loss.detach().item(),
                        drug_train_loss.detach().item(),
                        scalar_metric_value(fg_train_loss),
                        auc_loss.detach().item()
                    )
                )
                continue
            train_losses_in_epoch.append(loss.item())
            train_main_losses_in_epoch.append(train_loss.detach().item())
            train_local_losses_in_epoch.append(drug_train_loss.item())
            train_global_losses_in_epoch.append(scalar_metric_value(fg_train_loss))
            train_auc_losses_in_epoch.append(auc_loss.detach().item())
            train_mol_aux_losses_in_epoch.append(scalar_metric_value(mol_aux_loss))
            train_sparse_losses_in_epoch.append(scalar_metric_value(sparse_loss))
            train_shortcut_losses_in_epoch.append(scalar_metric_value(shortcut_loss))
            train_calib_losses_in_epoch.append(calibration_loss.detach().item())
            train_gate_means_in_epoch.append(scalar_metric_value(mean_local_global_gate))
            train_evidence_norms_in_epoch.append(scalar_metric_value(evidence_norm))
            train_pair_score_means_in_epoch.append(scalar_metric_value(pair_score_mean))
            train_motif_entropy_in_epoch.append(scalar_metric_value(motif_attention_entropy))
            train_residue_entropy_in_epoch.append(scalar_metric_value(residue_attention_entropy))
            train_global_alpha_in_epoch.append(scalar_metric_value(global_alpha))
            train_perf_norms_in_epoch.append(scalar_metric_value(performance_feature_norm))
            train_fine_norms_in_epoch.append(scalar_metric_value(fine_feature_norm))
            train_coarse_norms_in_epoch.append(scalar_metric_value(coarse_feature_norm))
            train_atom_gate_in_epoch.append(scalar_metric_value(atom_pocket_gate_mean))
            train_fine_token_counts_in_epoch.append(scalar_metric_value(fine_target_count))
            loss.backward()
            clip_params = [p for p in model.parameters() if p.grad is not None]
            if mol_trainable:
                clip_params.extend([p for p in model_drug.parameters() if p.grad is not None])
            if clip_params:
                nn.utils.clip_grad_norm_(clip_params, max_norm=5.0)
            optimizer.step()
            scheduler.step()

        if not train_losses_in_epoch:
            raise RuntimeError("No valid training batches were processed in epoch {}".format(epoch))
        train_loss_a_epoch = np.average(train_losses_in_epoch)
        train_loss_main_epoch = np.average(train_main_losses_in_epoch)
        train_loss_d_epoch = np.average(train_local_losses_in_epoch)
        train_loss_f_epoch = np.average(train_global_losses_in_epoch)
        train_loss_auc_epoch = np.average(train_auc_losses_in_epoch)
        train_loss_mol_aux_epoch = np.average(train_mol_aux_losses_in_epoch)
        train_loss_sparse_epoch = np.average(train_sparse_losses_in_epoch)
        train_loss_shortcut_epoch = np.average(train_shortcut_losses_in_epoch)
        train_loss_calib_epoch = np.average(train_calib_losses_in_epoch)
        train_gate_mean_epoch = np.average(train_gate_means_in_epoch)
        train_evidence_norm_epoch = np.average(train_evidence_norms_in_epoch)
        train_pair_score_mean_epoch = np.average(train_pair_score_means_in_epoch)
        train_motif_entropy_epoch = np.average(train_motif_entropy_in_epoch)
        train_residue_entropy_epoch = np.average(train_residue_entropy_in_epoch)
        train_global_alpha_epoch = np.average(train_global_alpha_in_epoch)
        train_perf_norm_epoch = np.average(train_perf_norms_in_epoch)
        train_fine_norm_epoch = np.average(train_fine_norms_in_epoch)
        train_coarse_norm_epoch = np.average(train_coarse_norms_in_epoch)
        train_atom_gate_epoch = np.average(train_atom_gate_in_epoch)
        train_fine_tokens_epoch = np.average(train_fine_token_counts_in_epoch)

        aux_loss_msg = (
            f"loss={train_loss_a_epoch:.5f} mol_stage={mol_stage} main={train_loss_main_epoch:.5f} "
            f"local_ce={train_loss_d_epoch:.5f} global_ce={train_loss_f_epoch:.5f} "
            f"auc_rank={train_loss_auc_epoch:.5f} "
            f"mol_aux={train_loss_mol_aux_epoch:.5f} sparse={train_loss_sparse_epoch:.5f} "
            f"shortcut={train_loss_shortcut_epoch:.5f} calib={train_loss_calib_epoch:.5f} "
            f"mean_gate={train_gate_mean_epoch:.5f} global_alpha={train_global_alpha_epoch:.5f} "
            f"perf_norm={train_perf_norm_epoch:.5f} fine_norm={train_fine_norm_epoch:.5f} "
            f"coarse_norm={train_coarse_norm_epoch:.5f} atom_gate={train_atom_gate_epoch:.5f} "
            f"fine_tokens={train_fine_tokens_epoch:.2f} evid_norm={train_evidence_norm_epoch:.5f} "
            f"pair_mean={train_pair_score_mean_epoch:.5f} motif_ent={train_motif_entropy_epoch:.5f} "
            f"res_ent={train_residue_entropy_epoch:.5f}"
        )

        valid_pbar = tqdm(
            enumerate(
                BackgroundGenerator(valid_dataset_load)),
            total=len(valid_dataset_load))
        valid_losses_in_epoch = []
        model.eval()
        model_drug.eval()
        Y, S = [], []
        dataloader_iterator = iter(valid_drug_load)
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:
                try:
                    batch = next(dataloader_iterator)
                except StopIteration:
                    break
                batch_graphs, num_atoms, edge_inputs, fg, fg_mask, \
                    in_degrees, out_degrees, spatial_pos, val_drug_idx, val_patch, val_patch_mask = batch

                batch_graphs = batch_graphs.to(device)
                val_num_atoms = num_atoms.to(device)
                in_degrees = in_degrees.to(device)
                out_degrees = out_degrees.to(device)
                edge_inputs = edge_inputs.to(device)
                spatial_pos = spatial_pos.to(device)
                fg = fg.to(device)
                fg_mask = fg_mask.to(device)
                val_patch = val_patch.to(device)
                val_patch_mask = val_patch_mask.to(device)

                val_pred = model_drug(
                    batch_graphs, edge_inputs, val_num_atoms, fg, fg_mask,
                    in_degrees, out_degrees, spatial_pos, val_patch, val_patch_mask,
                    return_aux=True
                )

                val_p1, val_p2, _, _ = unpack_drug_encoder_outputs(val_pred)
                valid_proteins, valid_labels, valid_drug_ids = valid_data
                assert_matching_drug_ids(valid_drug_ids, val_drug_idx, "valid", valid_i)
                valid_proteins = valid_proteins.to(device)
                valid_labels = valid_labels.to(device)

                outputs = model(val_p1, val_p2, val_num_atoms, valid_proteins, fg_mask)
                valid_logits, _, _, _ = unpack_model_outputs(outputs)
                valid_loss = Loss(valid_logits, valid_labels)

                valid_labels = valid_labels.detach().cpu().numpy()
                valid_scores = F.softmax(valid_logits, 1).detach().cpu().numpy()[:, 1]

                if not np.isfinite(valid_scores).all():
                    print("Skip non-finite validation batch {}".format(valid_i))
                    continue
                valid_losses_in_epoch.append(valid_loss.item())
                Y.extend(valid_labels)
                S.extend(valid_scores)

        if not Y:
            raise RuntimeError("No valid validation batches were processed in epoch {}".format(epoch))
        default_metrics = binary_metrics_from_scores(Y, S, threshold=0.5)
        should_refresh_calibration = (
            latest_calibration_result is None
            or epoch == 1
            or epoch % calibration_interval == 0
        )
        if should_refresh_calibration:
            calibration_result = find_best_temperature_and_threshold(Y, S, args)
            latest_calibration_result = calibration_result
        else:
            calibrated_scores = apply_temperature_to_scores(
                S, latest_calibration_result["temperature"]
            )
            val_threshold, threshold_metrics = find_best_threshold(
                Y, calibrated_scores, metric=args.threshold_metric
            )
            calibration_result = {
                "temperature": latest_calibration_result["temperature"],
                "threshold": val_threshold,
                "metrics": threshold_metrics,
                "scores": calibrated_scores,
            }
            latest_calibration_result = calibration_result
        val_temperature = calibration_result["temperature"]
        val_threshold = calibration_result["threshold"]
        threshold_metrics = calibration_result["metrics"]
        calibrated_scores = calibration_result["scores"]
        calibrated_default_metrics = binary_metrics_from_scores(Y, calibrated_scores, threshold=0.5)
        Reacll_dev = calibrated_default_metrics["recall"]
        Accuracy_dev = calibrated_default_metrics["accuracy"]
        AUC_dev = calibrated_default_metrics["auc"]
        PRC_dev = calibrated_default_metrics["prc"]
        MCC_dev = calibrated_default_metrics["mcc"]
        F1_dev = calibrated_default_metrics["f1"]

        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

        dev_score = validation_selection_score(
            AUC_dev, PRC_dev, threshold_metrics["mcc"], threshold_metrics["f1"],
            threshold_metrics["accuracy"], args)
        if is_better_auc_acc(dev_score, AUC_dev, threshold_metrics["accuracy"], best_score, best_auc, best_acc):
            best_drug_model = copy.deepcopy(model_drug)
            best_protein_model = copy.deepcopy(model)
            best_score = dev_score
            best_auc = AUC_dev
            best_acc = threshold_metrics["accuracy"]
            best_threshold = val_threshold
            best_temperature = val_temperature
            best_threshold_metrics = threshold_metrics

        epoch_len = len(str(num_epochs))

        print(
            f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] {aux_loss_msg} '
            f'val_loss={valid_loss_a_epoch:.5f} val_auc={AUC_dev:.5f} '
            f'val_aupr={PRC_dev:.5f} val_acc={Accuracy_dev:.5f} '
            f'val_recall={Reacll_dev:.5f} val_spec={default_metrics["specificity"]:.5f} '
            f'val_mcc={MCC_dev:.5f} val_f1={F1_dev:.5f} '
            f'cal_temp={val_temperature:.3f} cal_acc={calibrated_default_metrics["accuracy"]:.5f} '
            f'thr_metric={args.threshold_metric} val_best_thr={val_threshold:.3f} '
            f'val_best_mcc={threshold_metrics["mcc"]:.5f} '
            f'val_best_f1={threshold_metrics["f1"]:.5f} '
            f'select_score={dev_score:.5f} best_score={best_score:.5f} '
            f'best_auc={best_auc:.5f} best_acc={best_acc:.5f} '
            f'best_thr={best_threshold:.3f} best_temp={best_temperature:.3f}')

        early_stopping(-dev_score)
        if early_stopping.early_stop:
            print("Early stopping at epoch {} with best_score={:.5f}.".format(epoch, best_score))
            break

    if best_protein_model is None:
        best_protein_model = model
        best_drug_model = model_drug
    if best_threshold_metrics is not None:
        print(
            "Best validation threshold={:.3f}; temperature={:.3f}; val_acc={:.5f}; val_mcc={:.5f}; val_f1={:.5f}".format(
                best_threshold,
                best_temperature,
                best_threshold_metrics["accuracy"],
                best_threshold_metrics["mcc"],
                best_threshold_metrics["f1"]
            )
        )

    trainset_test_stable_results, _, _, _, _, _, _, _, _, _ = test_model(
        best_protein_model, best_drug_model, train_dataset_load, train_drug_load, save_path,
        dataset, Loss, device, dataset="Train", lable="stable",
        threshold=best_threshold, temperature=best_temperature
    )
    validset_test_stable_results, _, _, _, _, _, _, _, _, _ = test_model(
        best_protein_model, best_drug_model, valid_dataset_load, valid_drug_load, save_path,
        dataset, Loss, device, dataset="Valid", lable="stable",
        threshold=best_threshold, temperature=best_temperature
    )
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, F1_test, MCC_test, Sen_test, Spc_test = \
        test_model(
            best_protein_model, best_drug_model, test_dataset_load, test_drug_load, save_path,
            dataset, Loss, device, dataset="Test", lable="stable",
            threshold=best_threshold, temperature=best_temperature
        )
    export_test_evidence(
        best_protein_model,
        best_drug_model,
        test_dataset_load,
        test_drug_load,
        getattr(args, "export_evidence_path", ""),
        device,
        threshold=best_threshold,
        temperature=best_temperature,
        max_batches=getattr(args, "export_evidence_batches", 0)
    )
    AUC_List_stable.append(AUC_test)
    Accuracy_List_stable.append(Accuracy_test)
    AUPR_List_stable.append(PRC_test)
    Recall_List_stable.append(Recall_test)
    Precision_List_stable.append(Precision_test)
    F1_Score_List_stable.append(F1_test)
    MCC_List_stable.append(MCC_test)
    Sen_List_Stable.append(Sen_test)
    Spc_List_Stable.append(Spc_test)
    with open(save_path + "/The_results_of_whole_dataset.txt", 'a') as f:
        f.write("Test the stable model" + '\n')
        f.write(trainset_test_stable_results + '\n')
        f.write(validset_test_stable_results + '\n')
        f.write(testset_test_stable_results + '\n')

    show_result(dataset, train_type, Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                AUC_List_stable, AUPR_List_stable, F1_Score_List_stable, MCC_List_stable, Sen_List_Stable,
                Spc_List_Stable, setting_name=setting_name)

    current_time = timeit.default_timer()
    total_seconds = current_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = round(total_seconds % 60, 2)

    if hours > 0:
        print(f"训练总用时: {hours}小时 {minutes}分钟 {seconds}秒")
    elif minutes > 0:
        print(f"训练总用时: {minutes}分钟 {seconds}秒")
    else:
        print(f"训练总用时: {seconds}秒")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间：", current_time)


def main():
    args = parse_args()
    args.train_type = "transductive"
    assert args.gpu_id in range(0, 8)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    set_seed(config['seed'])

    cur_path = os.getcwd()
    data_path = osp.join(cur_path, "data")

    drug_dataset, train_dataset, valid_dataset, test_dataset, TVdata_len = get_data(data_path, args.dataset, config['seed'])
    run(args, config, device, drug_dataset, train_dataset, valid_dataset, test_dataset, TVdata_len,
        setting_name="transductive")


if __name__ == '__main__':
    main()
