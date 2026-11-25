# -*- coding: utf-8 -*-
# @Time     : 2025/04/08 上午9:11
# @Author   : joker
# @Email    : gryshr@163.com
# @File     : features.py
# @Project  : Graph_DPI

import os
import math
import torch
import numpy as np
import pickle
import os.path as osp
from datapreprocess.proteins import CustomDataset


cur_path = os.getcwd()
pro_path = osp.dirname(cur_path)
data_path = osp.join(pro_path, "data")

# DATASET = "DrugBank"
DATASET = "Davis"
# DATASET = "KIBA"
SEED = 42


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def save_dict(obj, name):
    file = osp.join(data_path, 'molecule_data', name + '.pkl')
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    print("Save done!")





def load_dataset():
    """
    load the dataset of interaction
    :return: data
    """
    print("Train in " + DATASET)
    file = osp.join(data_path, 'protein_data', DATASET)
    dir_input = ('{}.txt'.format(file))
    with open(dir_input, "r") as f:
        dataset = f.read().strip().split('\n')
    print("proteins load finished")

    # Get drug list.
    drug_list = []
    d_path = osp.join(data_path, DATASET + '_data_kg')
    for filename in os.listdir(d_path):
        name = osp.splitext(filename)[0]
        drug_list.append(name)

    # delete null data
    train_data = []
    for i in range(len(dataset)):
        pair = dataset[i]
        pair = pair.strip().split()
        drug_id, _, _, _ = pair[-4], pair[-3], pair[-2], pair[-1]
        if drug_id in drug_list:
            train_data.append(dataset[i])

    # random shuffle
    print("data shuffle")
    train_data = shuffle_dataset(train_data, SEED)
    return train_data


def generate_data(DATASET, data):
    data_len = len(data)
    drug_ids = []
    for i in range(data_len):
        pair = data[i]
        pair = pair.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
    drugs = []
    for v in range(data_len):
        filename = drug_ids[v]
        file = osp.join(data_path, DATASET + '_data_kg/', filename + '.pkl')
        d = load_dict(file)
        drugs.append(d)
    print(len(drugs), 'finished')
    return drugs


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data):

        self.graph_lists = []
        self.num_atoms = []
        self.in_degree = []
        self.out_degree = []
        self.spatial_pos = []
        self.edge_input = []
        self.edge_lists = []
        self.edge_feature = []
        self.atom_type = []
        self.smiles = []
        self.n_samples = len(data)
        self.mol_graph_list = []
        self.node_feature = []
        self.drug_id = []

        self.fg_list = []
        self.num_fg = []
        self.patch = []
        self.edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]
        self._prepare(data)

    def _prepare(self, data):
        for molecule in data:
            self.drug_id.append(molecule['id'])

    def __len__(self):
        """ Return the number of graphs in the dataset. """
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.graph_lists[idx],
            self.num_atoms[idx],
            self.edge_input[idx],
            self.fg_list[idx],
            self.num_fg[idx],
            self.in_degree[idx],
            self.out_degree[idx],
            self.spatial_pos[idx],
            self.drug_id[idx],
            self.patch[idx],
        )


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, train, valid, test):

        self.train = MoleculeDGL(train)
        print('train dataset finished')

        self.val = MoleculeDGL(valid)
        print('valid dataset finished')

        self.test = MoleculeDGL(test)
        print('test dataset finished')

        print('Error mol smiles:')
        # print(self.test.error_dict)


def load_dict(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def main():
    dataset = load_dataset()
    TVdataset_len = math.ceil(0.8 * len(dataset))
    torch.manual_seed(SEED)
    TV_data, test_data = torch.utils.data.random_split(dataset, [TVdataset_len, len(dataset) - TVdataset_len])

    TVdata, testdata = [], []
    for i in range(len(TV_data)):
        pair = dataset[TV_data.indices[i]]
        TVdata.append(pair)
    for i in range(len(test_data)):
        pair = dataset[test_data.indices[i]]
        testdata.append(pair)

    valid_size = math.ceil(0.2 * len(TVdata))
    torch.manual_seed(SEED)
    train_data, valid_data = torch.utils.data.random_split(TVdata, [len(TVdata) - valid_size, valid_size])

    traindata, validdata = [], []
    for i in range(len(train_data)):
        pair = TVdata[train_data.indices[i]]
        traindata.append(pair)
    for i in range(len(valid_data)):
        pair = TVdata[valid_data.indices[i]]
        validdata.append(pair)

    drug_ids, protein_ids = [], []
    for pair in traindata:
        pair = pair.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)

    drug_set = list(set(drug_ids))
    drug_set.sort(key=drug_ids.index)
    protein_set = list(set(protein_ids))
    protein_set.sort(key=protein_ids.index)

    train_dataset, valid_dataset, test_dataset = [], [], []
    for i in range(len(testdata)):
        pair = testdata[i].strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        if drug_id in drug_set and protein_id in protein_set:
            test_dataset.append(testdata[i])

    train_dataset = CustomDataset(traindata)
    valid_dataset = CustomDataset(validdata)
    test_dataset = CustomDataset(test_dataset)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    """generate train, valid, test datasets"""
    train = generate_data(DATASET, train_dataset.pairs)
    valid = generate_data(DATASET, valid_dataset.pairs)
    test = generate_data(DATASET, test_dataset.pairs)

    datasets = MoleculeDatasetDGL(train, valid, test)
    print(len(datasets.train), len(datasets.val), len(datasets.test))

    data_dir = osp.join(data_path, 'molecule_data')
    if not osp.exists(data_dir):
        os.mkdir(data_dir)

    save_dict(datasets, DATASET + "_transductive_setting_" + str(SEED))


if __name__ == '__main__':
    main()
