# -*- coding: utf-8 -*-
import argparse
import gc
import os
import os.path as osp
import pickle

import esm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Offline ESM protein embedding precomputation.")
    parser.add_argument("--dataset", type=str, default="KIBA", choices=["Davis", "KIBA", "DrugBank"])
    parser.add_argument("--esm_model", type=str, default="esm2_t33_650M_UR50D",
                        choices=["esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D", "esm2_t12_35M_UR50D"])
    parser.add_argument("--batch_size", type=int, default=8,
                        help="number of proteins per batch; real throughput is limited by token budget")
    parser.add_argument("--tokens_per_batch", type=int, default=6000,
                        help="upper bound on total amino-acid tokens per batch")
    parser.add_argument("--repr_layer", type=int, default=-1,
                        help="ESM representation layer; -1 means the last layer of the selected model")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="number of final ESM hidden layers to cache when pooling=per_token")
    parser.add_argument("--max_length", type=int, default=1022,
                        help="maximum sequence length accepted by ESM2")
    parser.add_argument("--truncate_mode", type=str, default="center", choices=["center", "head", "tail"],
                        help="how to crop sequences longer than max_length")
    parser.add_argument("--pooling", type=str, default="per_token", choices=["mean", "cls", "per_token"],
                        help="save mean pooled vector, cls vector, or per-token representation")
    parser.add_argument("--output", type=str, default="",
                        help="optional output path; default is data/protein_data/<dataset>_esm_<model>.pkl")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even if output cache already contains the protein id")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def get_project_data_root():
    cur_path = os.getcwd()
    local_data = osp.join(cur_path, "data")
    if osp.isdir(local_data):
        return local_data
    pro_path = osp.dirname(cur_path)
    parent_data = osp.join(pro_path, "data")
    if osp.isdir(parent_data):
        return parent_data
    raise FileNotFoundError("Cannot locate data directory from cwd={}".format(cur_path))


def load_dataset_lines(data_root, dataset):
    file_prefix = osp.join(data_root, "protein_data", dataset)
    txt_path = "{}.txt".format(file_prefix)
    with open(txt_path, "r") as handle:
        return handle.read().strip().split("\n")


def extract_unique_proteins(lines):
    protein_dict = {}
    for line in lines:
        pair = line.strip().split()
        _, protein_id, protein_seq, _ = pair[-4], pair[-3], pair[-2], pair[-1]
        protein_dict.setdefault(protein_id, protein_seq)
    return protein_dict


def truncate_sequence(sequence, max_length, mode):
    if len(sequence) <= max_length:
        return sequence
    if mode == "head":
        return sequence[:max_length]
    if mode == "tail":
        return sequence[-max_length:]
    start = max(0, (len(sequence) - max_length) // 2)
    return sequence[start:start + max_length]


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_esm_model(model_name, device):
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        last_layer = 33
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        last_layer = 30
    else:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        last_layer = 12
    model = model.to(device)
    model.eval()
    return model, alphabet, last_layer


def load_existing_cache(output_path):
    if not osp.exists(output_path):
        return {}
    with open(output_path, "rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise ValueError("Existing cache must be a dict keyed by protein_id: {}".format(output_path))
    return cache


def build_batches(protein_items, tokens_per_batch, batch_size):
    items = sorted(protein_items, key=lambda item: len(item[1]))
    batches = []
    current = []
    current_tokens = 0
    for item in items:
        seq_len = len(item[1])
        if current and (len(current) >= batch_size or current_tokens + seq_len > tokens_per_batch):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += seq_len
    if current:
        batches.append(current)
    return batches


def pool_representation(representations, tokens, alphabet, mode):
    if mode == "per_token":
        return representations[1:-1].cpu()
    if mode == "cls":
        return representations[0].cpu()
    valid_mask = (tokens != alphabet.padding_idx) & (tokens != alphabet.cls_idx) & (tokens != alphabet.eos_idx)
    if valid_mask.any():
        return representations[valid_mask].mean(dim=0).cpu()
    return representations[0].cpu()


def save_cache(cache, output_path):
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, output_path)


def main():
    args = parse_args()
    data_root = get_project_data_root()
    output_path = args.output or osp.join(
        data_root,
        "protein_data",
        "{}_esm_{}.pkl".format(args.dataset, args.esm_model)
    )

    lines = load_dataset_lines(data_root, args.dataset)
    proteins = extract_unique_proteins(lines)
    truncated_items = [
        (protein_id, truncate_sequence(sequence, args.max_length, args.truncate_mode))
        for protein_id, sequence in proteins.items()
    ]

    cache = load_existing_cache(output_path)
    if args.overwrite:
        pending_items = truncated_items
    else:
        pending_items = [(protein_id, seq) for protein_id, seq in truncated_items if protein_id not in cache]

    print("dataset={} unique_proteins={} cached={} pending={}".format(
        args.dataset, len(truncated_items), len(cache), len(pending_items)
    ))
    if args.pooling == "per_token":
        print("Warning: current EvidenceDTI integration expects pooled 1D ESM vectors; per_token cache is not supported.")

    if not pending_items:
        print("No pending proteins. Cache is up to date: {}".format(output_path))
        return

    device = resolve_device(args.device)
    model, alphabet, last_layer = load_esm_model(args.esm_model, device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = last_layer if args.repr_layer < 0 else int(args.repr_layer)
    repr_layers = [repr_layer]
    if args.pooling == "per_token":
        num_layers = max(1, int(args.num_layers))
        repr_layers = list(range(max(1, repr_layer - num_layers + 1), repr_layer + 1))
    batches = build_batches(pending_items, args.tokens_per_batch, args.batch_size)

    print("device={} esm_model={} repr_layers={} pooling={} batches={}".format(
        device, args.esm_model, repr_layers, args.pooling, len(batches)
    ))

    with torch.no_grad():
        for batch_index, batch in enumerate(batches, start=1):
            batch_labels, _, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device, non_blocking=True)
            outputs = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)

            for sample_index, protein_id in enumerate(batch_labels):
                if args.pooling == "per_token":
                    layer_stack = []
                    for layer_id in repr_layers:
                        layer_stack.append(
                            pool_representation(
                                outputs["representations"][layer_id][sample_index],
                                batch_tokens[sample_index],
                                alphabet,
                                args.pooling
                            )
                        )
                    cache[protein_id] = torch.stack(layer_stack, dim=0)
                else:
                    cache[protein_id] = pool_representation(
                        outputs["representations"][repr_layer][sample_index],
                        batch_tokens[sample_index],
                        alphabet,
                        args.pooling
                    )

            print("processed {}/{} batches; cache_size={}".format(
                batch_index, len(batches), len(cache)
            ))
            save_cache(cache, output_path)

            del batch_tokens, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    print("Saved cache to {}".format(output_path))


if __name__ == "__main__":
    main()
