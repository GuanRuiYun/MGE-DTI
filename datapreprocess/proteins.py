# -*- coding: utf-8 -*-
import os.path as osp
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def label_sequence(line, smi_ch_ind, max_seq_len=1000):
    encoded = np.zeros(max_seq_len, np.int64())
    for i, ch in enumerate(line[:max_seq_len]):
        encoded[i] = smi_ch_ind[ch]
    return encoded


class CustomDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


class ProteinBatch(dict):
    def to(self, device):
        moved = {}
        for key, value in self.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return ProteinBatch(moved)


def _default_esm_cache_path(data_root, dataset, esm_model):
    return osp.join(data_root, "protein_data", "{}_esm_{}.pkl".format(dataset, esm_model))


def _load_esm_cache(path):
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise ValueError("ESM cache must be a dict keyed by protein_id: {}".format(path))
    normalized = {}
    for protein_id, value in cache.items():
        if torch.is_tensor(value):
            normalized[str(protein_id)] = value.float()
        else:
            normalized[str(protein_id)] = torch.as_tensor(value, dtype=torch.float32)
    return normalized


def create_collate_fn(return_drug_ids=False,
                      protein_input_mode="esm",
                      data_root=None,
                      dataset=None,
                      esm_model="esm2_t33_650M_UR50D",
                      protein_max=1000):
    protein_input_mode = str(protein_input_mode).lower()
    if protein_input_mode not in {"esm", "raw"}:
        raise ValueError("protein_input_mode must be one of {'esm', 'raw'}, got {}".format(protein_input_mode))

    esm_cache = None
    if protein_input_mode == "esm":
        if data_root is None or dataset is None:
            raise ValueError("data_root and dataset are required when protein_input_mode='esm'.")
        esm_path = _default_esm_cache_path(data_root, dataset, esm_model)
        esm_cache = _load_esm_cache(esm_path)

    def collate_fn(batch):
        batch_size = len(batch)
        labels_new = torch.zeros(batch_size, dtype=torch.long)
        drug_ids = []

        if protein_input_mode == "raw":
            raw_tokens = torch.zeros((batch_size, protein_max), dtype=torch.long)
        else:
            esm_features = []
            esm_lengths = torch.zeros(batch_size, dtype=torch.long)
            esm_layer_counts = torch.zeros(batch_size, dtype=torch.long)

        for i, pair in enumerate(batch):
            pair = pair.strip().split()
            drug_id, protein_id, protein_str, label = pair[-4], pair[-3], pair[-2], pair[-1]
            drug_ids.append(drug_id)
            labels_new[i] = np.int32(label)

            if protein_input_mode == "raw":
                protein_int = torch.from_numpy(label_sequence(protein_str, CHARPROTSET, protein_max))
                raw_tokens[i] = protein_int
            else:
                if protein_id not in esm_cache:
                    raise KeyError("Protein id {} is missing from ESM cache.".format(protein_id))
                feature = esm_cache[protein_id]
                if feature.dim() == 1:
                    feature = feature.unsqueeze(0)
                if feature.dim() == 2:
                    feature = feature.unsqueeze(0)
                if feature.dim() != 3:
                    raise ValueError("ESM feature must be 1D pooled, 2D per-token, or 3D multi-layer per-token, got {} for protein {}."
                                     .format(tuple(feature.shape), protein_id))
                esm_features.append(feature)
                esm_layer_counts[i] = feature.size(0)
                esm_lengths[i] = feature.size(1)

        if protein_input_mode == "raw":
            protein_batch = ProteinBatch({
                "input_type": "raw",
                "raw_tokens": raw_tokens
            })
        else:
            max_layers = int(esm_layer_counts.max().item()) if batch_size > 0 else 1
            max_length = int(esm_lengths.max().item()) if batch_size > 0 else 1
            feature_dim = int(esm_features[0].size(-1)) if esm_features else 1
            protein_tensor = torch.zeros((batch_size, max_layers, max_length, feature_dim), dtype=torch.float32)
            for i, feature in enumerate(esm_features):
                num_layers, seq_len, hidden_dim = feature.shape
                protein_tensor[i, :num_layers, :seq_len, :hidden_dim] = feature
            protein_batch = ProteinBatch({
                "input_type": "esm",
                "esm_embedding": protein_tensor,
                "esm_lengths": esm_lengths,
                "esm_layer_counts": esm_layer_counts
            })

        if return_drug_ids:
            return protein_batch, labels_new, drug_ids
        return protein_batch, labels_new

    return collate_fn
