# -*- coding: utf-8 -*-
# @Time     : 2025-04-10 20:52
# @Author   : joker
# @Email    : gryshr@163.com
# @File     : add_smiles_for_pkl.py
# @Project  : Graph_DTI

import os
import dgl
import json
import torch
import metis
import pickle
import numpy as np
import networkx as nx
import os.path as osp
from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict
import torch.nn.functional as F
from datapreprocess.features import match_fg, getFragmentFeats, getBricsBonds, atom_features, bond_features

cur_path = os.getcwd()
pro_path = osp.dirname(cur_path)
data_path = osp.join(pro_path, "data")
edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]


def load_txt(filename):
    with open("%s.txt" % (filename), "r") as f:
        data = f.read().strip().split("\n")

    data = json.loads(data[0])
    return data


def load_pkl(path):
    data = dict()
    for filename in os.listdir(path):
        full_path = osp.join(path, filename)  # 拼接绝对路径
        name = osp.splitext(filename)[0]
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
        data[name] = obj

    return data


def save_pkl(filename, p_data):
    with open(filename+".pkl", 'wb') as f:  # 必须用二进制写入模式
        pickle.dump(p_data, f)  # 序列化对象到文件‌


def load_dict(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def metis_partition(g, n_patches=50):
    if g['num_nodes'] < n_patches:
        membership = torch.randperm(n_patches)
    else:
        # data augmentation
        # adjlist = g['edge_index'].t()
        adjlist = g['edge_index']
        G = nx.Graph()
        G.add_nodes_from(np.arange(g['num_nodes']))
        G.add_edges_from(adjlist.tolist())
        # metis partition
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= g['num_nodes']
    membership = torch.tensor(membership[:g['num_nodes']])

    patch = []
    max_patch_size = 15
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [g['num_nodes']] * (max_patch_size - l)

    patch = torch.tensor(patch)

    return patch


def build_mol_data(id, smiles, entity_data, raw_data):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    num_patch = 50

    num_fg = []
    fg_list, fg_len = match_fg(mol)
    result_ap, result_p = getFragmentFeats(mol)
    reaction_idx, bbr = getBricsBonds(mol)

    # 直接一步生成分组集合列表
    groups = defaultdict(list)
    for atom_key, group_id in result_ap.items():
        groups[group_id].append(atom_key)

    # 将每个分组转为set，并生成列表
    frag_groups = [atom_keys for atom_keys in groups.values()]

    num_fg.append([fg_len, frag_groups])

    atom_type = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    atom_feature = [torch.tensor(atom_features(mol.GetAtomWithIdx(i))) for i in range(mol.GetNumAtoms())]

    num_atoms = mol.GetNumAtoms()

    assert num_atoms == raw_data['num_atom'], f"The atom number not equal. The smiles is {smiles}, The id is {id}."

    atom_ele_feature = [entity_data[atom] for atom in atom_type]
    mol_atom_feature = [torch.cat((t1, t2), dim=0) for t1, t2 in zip(atom_feature, atom_ele_feature)]   # 128+133=261
    # out_degree = defaultdict(int)
    # in_degree = defaultdict(int)
    homo_edge = list()
    edges = {k: [] for k in edge_types}
    for bond in mol.GetBonds():
        edges[('a', 'b', 'a')].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[('a', 'b', 'a')].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        homo_edge.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        homo_edge.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        # out_degree[bond.GetBeginAtomIdx()] += 1
        # in_degree[bond.GetEndAtomIdx()] += 1

    # mol BRICS bonds disconnect, bond of connect the two fragment
    for r in reaction_idx:
        begin = r[1]
        end = r[2]
        edges[('p', 'r', 'p')].append([result_ap[begin], result_ap[end]])
        edges[('p', 'r', 'p')].append([result_ap[end], result_ap[begin]])

    # atom and fragment relationships
    for k, v in result_ap.items():
        edges[('a', 'j', 'p')].append([k, v])  # atom -> fragment
        edges[('p', 'j', 'a')].append([v, k])  # fragment -> atom

    mol_g = dgl.heterograph(edges)
    mol_atom_feature = torch.stack(mol_atom_feature, dim=0)
    mol_g.nodes['a'].data['f'] = mol_atom_feature

    f_pharm = []
    for k, v in result_p.items():
        f_pharm.append(v)
    mol_g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    dim_atom = len(mol_atom_feature[0])
    dim_atom_padding = mol_g.nodes['a'].data['f'].size()[0]  # number of atoms
    dim_pharm_padding = mol_g.nodes['p'].data['f'].size()[0]  # number of fragments

    mol_g.nodes['a'].data['f_junc'] = torch.cat(
        [mol_g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)],
        1)
    mol_g.nodes['p'].data['f_junc'] = torch.cat(
        [torch.zeros(dim_pharm_padding, dim_atom), mol_g.nodes['p'].data['f']],
        1)

    # features of edges
    f_bond = []
    src, dst = mol_g.edges(etype=('a', 'b', 'a'))
    for i in range(mol_g.num_edges(etype=('a', 'b', 'a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    mol_g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)

    # fragments relationship feature
    f_reac = []
    src, dst = mol_g.edges(etype=('p', 'r', 'p'))
    for idx in range(mol_g.num_edges(etype=('p', 'r', 'p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    mol_g.edges[('p', 'r', 'p')].data['x'] = torch.FloatTensor(f_reac)  # fragment relationship feature
    mol_data = defaultdict()
    mol_data['id'] = id
    mol_data['fg_list'] = fg_list
    mol_data['num_atom'] = num_atoms

    mol_data['mol_graph'] = mol_g
    mol_data['num_fg'] = num_fg

    raw_node_feature = raw_data['atom_type'].long()

    spatial_pos_list = (raw_data['spatial_pos'] != 0).nonzero()
    spatial = torch.stack([raw_data['spatial_pos'][src, dst] for src, dst in spatial_pos_list])

    adj = torch.sum(raw_data['edge_input'], dim=2)
    edge_list = (adj != 0).nonzero()

    # Add patch split 20251207
    patch_g = dict()
    patch_g['num_nodes'] = mol_data['num_atom']
    patch_g['edge_index'] = edge_list

    patch = metis_partition(patch_g, num_patch)
    mol_data['num_atom'] += 1
    raw_node_feature = F.pad(raw_node_feature, [0, 1])


    # Create the DGL Graph
    g = dgl.DGLGraph()
    g.add_nodes(mol_data['num_atom'])
    g.ndata['feat'] = raw_node_feature

    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    edge = torch.stack([raw_data['edge_input'][src, dst, :] for src, dst in edge_list]).long()

    # raw data
    # mol_data['atom_type'] = raw_data['atom_type']
    if raw_data['in_degree'][-1] != 0:
        raw_data['in_degree'].append(0)

    if raw_data['out_degree'][-1] != 0:
        raw_data['out_degree'].append(0)

    mol_data['in_degree'] = raw_data['in_degree']
    mol_data['out_degree'] = raw_data['out_degree']
    mol_data['spatial_pos'] = spatial
    mol_data['edge_list'] = edge_list
    mol_data['edge_input'] = edge
    mol_data['graph'] = g
    mol_data['patch'] = patch
    return mol_data


def align_dataset_smiles(dataset):
    f_smile_path = osp.join(data_path, dataset, "ligands_can")
    f_raw_path = osp.join(data_path, dataset+"_drugs")
    smile_data = load_txt(f_smile_path)
    # print(type(smile_data), len(smile_data))        # len 68
    raw_mol_data = load_pkl(f_raw_path)
    # print(type(raw_mol_data), len(raw_mol_data))    # len 67
    # smile_data_keys = smile_data.keys()
    # raw_keys = raw_mol_data.keys()

    # print(sorted(smile_data_keys))
    # print(sorted(raw_keys))

    common_keys = list(smile_data.keys() & raw_mol_data.keys())
    if dataset == 'Davis':
        common_keys.remove('44150621')
        common_keys.remove('9809715')
    elif dataset == 'KIBA':
        common_keys.remove('CHEMBL1163016')

    # print(raw_mol_data['10074640']) # Davis
    # emb_file = osp.join(data_path, 'kg', 'RotatE_128_64_emb.pkl')
    raw_file = osp.join(data_path, 'kg', 'RotatE_128_64.pkl')

    # emb_d = load_dict(emb_file)
    raw_d = load_dict(raw_file)

    entity_ele = {v: k for k, v in raw_d['id2entity'].items()}
    entity_emb = {k: raw_d['entity'][v] for k, v in entity_ele.items()}

    # Generate file dir
    res_dir = dataset + "_data_kg"
    dir_path = osp.join(data_path, res_dir)

    if not osp.exists(dir_path):
        os.mkdir(dir_path)

    for k in tqdm(common_keys):
        gen_data = build_mol_data(k, smile_data[k], entity_emb, raw_mol_data[k])

        files = osp.join(dir_path, k)
        save_pkl(files, gen_data)

    print("处理完成！")


def main():
    # dataset = "Davis"
    dataset = "KIBA"
    align_dataset_smiles(dataset)


if __name__ == '__main__':
    main()