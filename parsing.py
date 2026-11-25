# -*- coding: utf-8 -*-
# @Time     : 2025/4/14 下午8:34
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : parsing.py
# @Project  : Graph_DTI
# @Description :
import argparse


def parse_args():
    """Parses arguments for MGE-DTI (EvidenceDTI + MolCoBFormer) training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Davis')  # Davis, KIBA, DrugBank
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--train_type', type=str, default='transductive')  # transductive, drug_inductive, drug_protein_inductive
    parser.add_argument('--num_works', type=int, default=0)

    parser.add_argument('--unfreeze_molcobformer', action='store_true',
                        help='fine-tune MolCoBFormer instead of using it as a frozen feature extractor')
    parser.add_argument('--mol_lr_mult', type=float, default=0.03,
                        help='MolCoBFormer learning-rate multiplier when MolCoBFormer is unfrozen')
    parser.add_argument('--mol_unfreeze_strategy', type=str, default='staged',
                        choices=['none', 'staged', 'full'],
                        help='MolCoBFormer unfreeze schedule; --unfreeze_molcobformer maps to full')
    parser.add_argument('--mol_stage1_epochs', type=int, default=8,
                        help='epochs to keep MolCoBFormer frozen when --mol_unfreeze_strategy staged')
    parser.add_argument('--mol_stage2_epochs', type=int, default=20,
                        help='last epoch of FG/local/last-BGA partial unfreeze when --mol_unfreeze_strategy staged')

    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='override config patience when > 0')
    parser.add_argument('--model_name', type=str, default='evidence',
                        choices=['evidence'],
                        help='EvidenceDTI mainline model')

    parser.add_argument('--main_loss_weight', type=float, default=1.0,
                        help='main fused-logit classification loss weight')
    parser.add_argument('--local_loss_weight', type=float, default=0.25,
                        help='auxiliary motif-residue local-evidence classification loss weight')
    parser.add_argument('--global_loss_weight', type=float, default=0.10,
                        help='auxiliary coarse/global classification loss weight used by EvidenceDTI')
    parser.add_argument('--pairwise_auc_weight', type=float, default=0.12,
                        help='non-negative pairwise ranking loss weight for fused positive scores')
    parser.add_argument('--ranking_margin', type=float, default=0.8,
                        help='margin used by the pairwise ranking loss')
    parser.add_argument('--ranking_hard_topk_frac', type=float, default=0.35,
                        help='top fraction of hardest pairwise ranking terms to keep, in (0, 1]')
    parser.add_argument('--mol_aux_loss_weight', type=float, default=0.0,
                        help='MolCoBFormer locality/motif auxiliary loss weight when it is unfrozen')
    parser.add_argument('--mol_align_loss_weight', type=float, default=0.0,
                        help='explicit MolCoBFormer topology/message-passing alignment loss weight')
    parser.add_argument('--evidence_sparsity_weight', type=float, default=0.0,
                        help='weight for top-k evidence sparsity regularization')
    parser.add_argument('--shortcut_loss_weight', type=float, default=0.0,
                        help='weight for global shortcut uniformity penalty')
    parser.add_argument('--calibration_loss_weight', type=float, default=0.0,
                        help='weight for Brier calibration loss')

    parser.add_argument('--evidence_topk', type=int, default=24,
                        help='number of motif-residue pairs used as local evidence')
    parser.add_argument('--pocket_slots', type=int, default=16,
                        help='latent protein pocket slots used when structure is incomplete')
    parser.add_argument('--pocket_refine_layers', type=int, default=0,
                        help='number of lightweight transformer layers used to refine latent pocket tokens')
    parser.add_argument('--pocket_refine_heads', type=int, default=4,
                        help='attention heads used by the latent pocket token refiner')
    parser.add_argument('--pocket_refine_dropout', type=float, default=0.1,
                        help='dropout used by the latent pocket token refiner')
    parser.add_argument('--residue_dropout', type=float, default=0.05,
                        help='modality dropout over residue tokens for structure-optional training')
    parser.add_argument('--global_calibration_max', type=float, default=0.05,
                        help='maximum contribution scale of global calibration logits')
    parser.add_argument('--evidence_aux_logit_weight', type=float, default=0.0,
                        help='direct logit contribution of the sparse evidence head; keep 0 for performance-first runs')
    parser.add_argument('--fine_interaction_target', type=str, default='pocket',
                        choices=['pocket', 'residue', 'candidate', 'topk_residue'],
                        help='protein tokens used by the EvidenceDTI atom-level performance branch')
    parser.add_argument('--fine_residue_topk', type=int, default=128,
                        help='number of pocket-attended residues used when --fine_interaction_target topk_residue')
    parser.add_argument('--performance_interaction_heads', type=int, default=4,
                        help='attention heads used by the EvidenceDTI coarse-to-fine performance branch')
    parser.add_argument('--performance_interaction_dropout', type=float, default=0.1,
                        help='dropout used by the EvidenceDTI coarse-to-fine performance branch')
    parser.add_argument('--evidence_context_layers', type=int, default=2,
                        help='number of lightweight Transformer layers used to contextualize evidence tokens in EvidenceDTI')
    parser.add_argument('--evidence_context_heads', type=int, default=4,
                        help='number of attention heads used by the evidence context encoder in EvidenceDTI')
    parser.add_argument('--evidence_attention', type=str, default='sparsemax',
                        choices=['sparsemax', 'softmax'],
                        help='masked attention normalizer used by the sparse evidence aggregator')
    parser.add_argument('--prediction_mode', type=str, default='full',
                        choices=['full', 'performance_only', 'local_only', 'global_only'],
                        help='final-logit path used for ablation: full, performance branch only, local evidence only, or global branch only')
    parser.add_argument('--disable_coarse_context', action='store_true',
                        help='remove the coarse motif-pocket context from the atom-level fine interaction gate')
    parser.add_argument('--export_evidence_path', type=str, default='',
                        help='optional CSV path for exporting top-k test-set evidence after training')
    parser.add_argument('--export_evidence_batches', type=int, default=0,
                        help='maximum number of test batches to export evidence from; 0 exports the full test split')

    parser.add_argument('--protein_input_mode', type=str, default='esm',
                        choices=['esm', 'raw'],
                        help='protein representation source; esm loads offline ESM cache, raw uses integer amino-acid sequence tokens')
    parser.add_argument('--esm_model_name', type=str, default='esm2_t33_650M_UR50D',
                        choices=['esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D'],
                        help='offline ESM cache model name used to resolve protein_data/<dataset>_esm_<model>.pkl')
    parser.add_argument('--esm_embedding_dim', type=int, default=1280,
                        help='dimension of pooled ESM protein embeddings')
    parser.add_argument('--esm_num_layers', type=int, default=4,
                        help='number of cached ESM hidden layers used for learnable layer mixing in EvidenceDTI')
    parser.add_argument('--protein_adapter_layers', type=int, default=2,
                        help='number of lightweight transformer layers used to adapt ESM residue embeddings before evidence matching')

    parser.add_argument('--selection_metric', type=str, default='balanced',
                        choices=['balanced', 'auc_acc'],
                        help='validation model-selection score: balanced uses AUC/AUPR/MCC/F1; auc_acc uses AUC/ACC')
    parser.add_argument('--threshold_metric', type=str, default='f1_mcc',
                        choices=['f1_mcc', 'mcc', 'f1', 'youden'],
                        help='validation threshold selection metric used for final train/valid/test evaluation')
    parser.add_argument('--temperature_min', type=float, default=0.7,
                        help='minimum temperature used during validation calibration search')
    parser.add_argument('--temperature_max', type=float, default=1.6,
                        help='maximum temperature used during validation calibration search')
    parser.add_argument('--temperature_grid_size', type=int, default=19,
                        help='number of temperatures used during validation calibration search')
    parser.add_argument('--temperature_update_interval', type=int, default=5,
                        help='run validation temperature search every N epochs; other epochs reuse the last best temperature')
    parser.add_argument('--auc_weight', type=float, default=0.4,
                        help='AUC weight for validation model-selection score')
    parser.add_argument('--aupr_weight', type=float, default=0.3,
                        help='AUPR weight when --selection_metric balanced')
    parser.add_argument('--mcc_weight', type=float, default=0.2,
                        help='validation MCC weight when --selection_metric balanced')
    parser.add_argument('--f1_weight', type=float, default=0.1,
                        help='validation F1 weight when --selection_metric balanced')
    parser.add_argument('--class_weight_strategy', type=str, default='default',
                        choices=['default', 'none', 'balanced', 'custom'],
                        help='class weighting for cross entropy: default keeps legacy dataset weights, none disables weights, balanced uses train split label frequencies, custom uses --class_weight_values')
    parser.add_argument('--class_weight_values', type=str, default='',
                        help='comma-separated class weights for --class_weight_strategy custom, e.g. 0.5,0.5')
    return parser.parse_args()
