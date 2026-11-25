# -*- coding: utf-8 -*-
# @Time     : 2025/4/14 下午8:50
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : get_data.py
# @Project  : Graph_DTI
# @Description :

import os
import math
import torch
import pickle
import numpy as np
import os.path as osp
from tqdm import trange
from datapreprocess.run import load_dict
from datapreprocess.proteins import CustomDataset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def load_pkl(path):
    data = dict()
    for filename in os.listdir(path):
        full_path = osp.join(path, filename)  # 拼接绝对路径
        name = osp.splitext(filename)[0]
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
        data[name] = obj

    return data


def load_dataset(data_path, dataset, seed):
    """
    load the dataset of interaction
    :return: data
    """
    print("Train in " + dataset)
    file = osp.join(data_path, 'protein_data', dataset)
    d_path = osp.join(data_path, dataset + '_data_kg')

    dir_input = ('{}.txt'.format(file))
    with open(dir_input, "r") as f:
        dataset = f.read().strip().split('\n')
    print("proteins load finished")
    # Get drug list.
    drug_list = []
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
    train_data = shuffle_dataset(train_data, seed)
    return train_data


def get_data(path, name, seed):
    file = osp.join(path, 'molecule_data', name + '_transductive_setting_' + str(seed) + '.pkl')
    # preprocess the data that is generated graph
    data = load_dict(file)

    dataset = load_dataset(path, name, seed)
    # load the molecular data
    mol_path = osp.join(path, name+'_data_kg')
    mol_data = load_pkl(mol_path)

    # split the data for train, valid and test.
    TVdataset_len = math.ceil(0.8 * len(dataset))
    torch.manual_seed(seed)
    TV_data, test_data = torch.utils.data.random_split(dataset, [TVdataset_len, len(dataset) - TVdataset_len])

    TVdata, testdata = [], []
    for i in range(len(TV_data)):
        pair = dataset[TV_data.indices[i]]
        TVdata.append(pair)
    for i in range(len(test_data)):
        pair = dataset[test_data.indices[i]]
        testdata.append(pair)

    valid_size = math.ceil(0.2 * len(TVdata))
    torch.manual_seed(seed)
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

    data = process_drug_data(data, mol_data)

    return data, train_dataset, valid_dataset, test_dataset, len(TVdata)


def get_inductive_data(path, name, seed):
    file = osp.join(path, 'molecule_data', name + '_drug_inductive_setting_' + str(seed) + '.pkl')
    # preprocess the data that is generated graph
    data = load_dict(file)

    dataset = load_dataset(path, name, seed)
    # load the molecular data
    mol_path = osp.join(path, name + '_data_kg')
    mol_data = load_pkl(mol_path)

    drug_ids = []
    for pair in dataset:
        pair = pair.strip().split()
        drug_id = pair[-4]
        drug_ids.append(drug_id)

    drug_set = list(set(drug_ids))
    drug_set.sort(key=drug_ids.index)

    torch.manual_seed(seed)
    train_d, test_d = torch.utils.data.random_split(drug_set, [math.ceil(0.8 * len(drug_set)),
                                                               len(drug_set) - math.ceil(0.8 * len(drug_set))])
    train_drug, test_drug = [], []
    for i in range(len(train_d)):
        pair = drug_set[train_d.indices[i]]
        train_drug.append(pair)
    for i in range(len(test_d)):
        pair = drug_set[test_d.indices[i]]
        test_drug.append(pair)

    TVdata, test_dataset = [], []
    for i in dataset:
        pair = i.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        if drug_id in train_drug:
            TVdata.append(i)
        if drug_id in test_drug:
            test_dataset.append(i)

    valid_size = math.ceil(0.2 * len(TVdata))
    torch.manual_seed(seed)
    train_data, valid_data = torch.utils.data.random_split(TVdata, [len(TVdata) - valid_size, valid_size])

    traindata, validdata = [], []
    for i in range(len(train_data)):
        pair = TVdata[train_data.indices[i]]
        traindata.append(pair)
    for i in range(len(valid_data)):
        pair = TVdata[valid_data.indices[i]]
        validdata.append(pair)

    train_dataset = CustomDataset(traindata)
    valid_dataset = CustomDataset(validdata)
    test_dataset = CustomDataset(test_dataset)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    data = process_drug_data(data, mol_data)

    return data, train_dataset, valid_dataset, test_dataset, len(TVdata)


def get_drug_protein_data(path, name, seed):
    file = osp.join(path, 'molecule_data', name + '_drug_protein_inductive_setting_' + str(seed) + '.pkl')
    # preprocess the data that is generated graph
    data = load_dict(file)

    dataset = load_dataset(path, name, seed)
    # load the molecular data
    mol_path = osp.join(path, name + '_data_kg')
    mol_data = load_pkl(mol_path)

    drug_ids, protein_ids = [], []
    for pair in dataset:
        pair = pair.strip().split()
        drug_id, protein_id = pair[-4], pair[-3]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)

    drug_set = list(set(drug_ids))
    drug_set.sort(key=drug_ids.index)
    protein_set = list(set(protein_ids))
    protein_set.sort(key=protein_ids.index)

    torch.manual_seed(seed)
    train_d, test_d = torch.utils.data.random_split(drug_set, [math.ceil(0.5 * len(drug_set)),
                                                               len(drug_set) - math.ceil(0.5 * len(drug_set))])
    train_drug, test_drug = [], []
    for i in range(len(train_d)):
        pair = drug_set[train_d.indices[i]]
        train_drug.append(pair)
    for i in range(len(test_d)):
        pair = drug_set[test_d.indices[i]]
        test_drug.append(pair)

    torch.manual_seed(seed)
    train_p, test_p = torch.utils.data.random_split(protein_set, [math.ceil(0.5 * len(protein_set)),
                                                                  len(protein_set) - math.ceil(0.5 * len(protein_set))])
    train_protein, test_protein = [], []
    for i in range(len(train_p)):
        pair = protein_set[train_p.indices[i]]
        train_protein.append(pair)
    for i in range(len(test_p)):
        pair = protein_set[test_p.indices[i]]
        test_protein.append(pair)

    TVdata, train_dataset, valid_dataset, test_dataset = [], [], [], []
    for i in dataset:
        pair = i.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        if drug_id in train_drug and protein_id in train_protein:
            TVdata.append(i)
        if drug_id in test_drug and protein_id in test_protein:
            test_dataset.append(i)

    valid_size = math.ceil(0.2 * len(TVdata))
    torch.manual_seed(seed)
    train_data, valid_data = torch.utils.data.random_split(TVdata, [len(TVdata) - valid_size, valid_size])

    traindata, validdata = [], []
    for i in range(len(train_data)):
        pair = TVdata[train_data.indices[i]]
        traindata.append(pair)
    for i in range(len(valid_data)):
        pair = TVdata[valid_data.indices[i]]
        validdata.append(pair)

    train_dataset = CustomDataset(traindata)
    valid_dataset = CustomDataset(validdata)
    test_dataset = CustomDataset(test_dataset)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    data = process_drug_data(data, mol_data)

    return data, train_dataset, valid_dataset, test_dataset, len(TVdata)


def process_drug_data(dataset, mol_data):
    train_data = dataset.train
    val_data = dataset.val
    test_data = dataset.test

    dataset.train = merge_mol_data(train_data, mol_data)
    dataset.val = merge_mol_data(val_data, mol_data)
    dataset.test = merge_mol_data(test_data, mol_data)

    return dataset


def merge_mol_data(data, mol_data):
    num_drug = data.n_samples
    for i in trange(num_drug):
        id = data.drug_id[i]
        data.fg_list.append(torch.tensor(mol_data[id]['fg_list']))
        data.num_atoms.append(mol_data[id]['num_atom'])
        data.num_fg.append(mol_data[id]['num_fg'][0])
        data.graph_lists.append(mol_data[id]['graph'])
        data.edge_input.append(mol_data[id]['edge_input'])
        data.in_degree.append(mol_data[id]['in_degree'])
        data.out_degree.append(mol_data[id]['out_degree'])
        data.spatial_pos.append(mol_data[id]['spatial_pos'])
        data.patch.append(mol_data[id]['patch'])

    return data
