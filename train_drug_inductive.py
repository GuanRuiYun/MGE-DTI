# -*- coding: utf-8 -*-
"""Cold-drug training entry for MGE-DTI.

This script intentionally reuses the current transductive training core and
changes only the data split. Keeping one training implementation avoids metric,
loss, calibration, and checkpoint-selection drift across experimental settings.
"""

import os
import os.path as osp

import torch
import yaml
from yaml import SafeLoader

from datapreprocess.get_data import get_inductive_data
from parsing import parse_args
from run_transductive import run, set_seed
from datapreprocess.run import MoleculeDatasetDGL, MoleculeDGL


def main():
    args = parse_args()
    args.train_type = "drug_inductive"
    assert args.gpu_id in range(0, 8)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    set_seed(config['seed'])

    data_path = osp.join(os.getcwd(), "data")
    drug_dataset, train_dataset, valid_dataset, test_dataset, TVdata_len = get_inductive_data(
        data_path,
        args.dataset,
        config['seed']
    )
    run(
        args,
        config,
        device,
        drug_dataset,
        train_dataset,
        valid_dataset,
        test_dataset,
        TVdata_len,
        setting_name="drug_inductive"
    )


if __name__ == '__main__':
    main()
