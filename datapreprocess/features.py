# -*- coding: utf-8 -*-
# @Time     : 2025/04/08 上午9:11
# @Author   : joker
# @Email    : gryshr@163.com
# @File     : features.py
# @Project  : Graph_DPI

import os
import copy
import torch
import numpy as np
import pickle
import csv
import itertools
from pathlib import Path
from argparse import Namespace
from typing import List, Tuple, Union, Any
from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig

from dhg import Hypergraph, Graph
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_scatter import scatter


# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2         # 133 = (100 + 1) + (5+1) * 5
BOND_FDIM = 14



# Memoization
SMILES_TO_GRAPH = {}

eletype_list = [i for i in range(118)]

# unknown generate 14 random features
hrc2emb = {}
for eletype in eletype_list:
    hrc_emb = np.random.rand(14)
    hrc2emb[eletype] = hrc_emb

# VSCode and PyCharm Path different
# VSCode can use cwd, The PyCharm use parent
# cur_path = Path.cwd()

# print(cur_path)
# parent_path = cur_path.parent

def _resolve_kg_data_path():
    env_path = os.environ.get("GRAPH_DTI_KG_DIR")
    file_path = Path(__file__).resolve()
    repo_root = file_path.parents[1]
    legacy_root = file_path.parents[2]

    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        repo_root.joinpath('data', 'kg'),
        repo_root.joinpath('code', 'data', 'kg'),
        legacy_root.joinpath('data', 'kg'),
        legacy_root.joinpath('code', 'data', 'kg'),
    ])

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.joinpath('ele2emb.pkl').exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate KG embedding directory. Checked: {}".format(
            ", ".join(str(candidate) for candidate in checked)
        )
    )


data_path = _resolve_kg_data_path()

# VSCode for train model and  train model
# data_path = cur_path.joinpath('data', 'kg')

# PyCharm for generate data use and generate the data. This is use for data embedding phase.
# data_path = parent_path.joinpath('data', 'kg')

# load elements embedding
ele2emb = pickle.load(open(data_path.joinpath('ele2emb.pkl'), 'rb'))

# load relation embedding
rel2emb = pickle.load(open(data_path.joinpath('rel2emb.pkl'), 'rb'))

# load functional groups embedding
fg2emb = pickle.load(open(data_path.joinpath('fg2emb.pkl'), 'rb'))

# load the smiles of functional groups
smart = []
smart2name = dict()
with open(data_path.joinpath('funcgroup.txt'), "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]
    smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
    smart2name = dict(zip(smart, name))


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """

    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def hrc_features(ele):
    fhrc = hrc2emb[ele]
    return fhrc.tolist()


def ele_features(ele):
    """
    Get element feature embedding

    :param ele: The element
    :return: The element feature
    """
    fele = ele2emb[ele]
    return fele.tolist()


def relation_features(e1, e2):
    """
    Get the relation feature embedding.

    :param e1: The src edge.
    :param e2: The dst edge.
    :return: relation feature
    """
    frel = rel2emb[(e1, e2)]
    return frel.tolist()


def match_fg(mol):
    """
    Match the functional groups of molecular

    :param mol: The molecular
    :return:
    """
    fg_emb = []
    pad_fg = [[0] * 133]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            fg_emb.append(fg2emb[smart2name[sm]].tolist())
    fg_len = len(fg_emb)
    if len(fg_emb) > 14:
        fg_emb = fg_emb[:14]
    else:
        fg_emb.extend(pad_fg * (14 - len(fg_emb)))

    # if len(fg_emb) == 0:
    #     fg_emb = [[0] * 133]
    return fg_emb, fg_len


def getFragmentFeats(mol):
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    result_ap = {}
    result_p = {}
    pharm_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            emb_0 = maccskeys_emb(mol_pharm)
            emb_1 = pharm_property_types_feats(mol_pharm)
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]

        result_p[pharm_id] = emb_0 + emb_1

        pharm_id += 1
    return result_ap, result_p


fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def pharm_property_types_feats(mol, factory=factory):
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result


def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))


def getBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]

    for item in bonds:  # item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], getBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], getBricsBondFeature([item[1][1], item[1][0]])])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])

    return result, brics_bonds_rules


def getBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] !='7a' and action[0] !='7b') else 7
    end_action_bond = int(action[1]) if (action[1] !='7a' and action[1] !='7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result


# build the mol graph
class MolGraph:
    r"""
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    Args:
        smiles: Smiles string.
        mol: The Molecular represent
    """

    def __init__(self, args, smiles: str, mol) -> None:
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.args = args
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.n_real_atoms = 0
        self.n_eles = 0
        self.bbr = []
        self.z = []

        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []

        # Convert smiles to molecule
        self.f_fgs = match_fg(mol)  # match the functional groups of smiles
        self.n_fgs = len(self.f_fgs)

        self.fragment_ap, self.f_fragment = getFragmentFeats(mol)
        assert len(self.fragment_ap) > 0, "fragment is null"
        if len(self.f_fragment) == 0:
            print("fragment:", len(self.f_fragment), self.fragment_ap)
        # self.reac_idx = getBricsBonds(mol)
        # self.reac_idx = sorted(self.reac_idx)
        # add get the
        # reference: atom concat pharm
        if args.train.use_geometric:
            # 获取原子3D坐标 positions
            self.coords = mol.GetConformer().GetPositions()
        else:
            self.coords = None
        # get the atom ID number in The Periodic Table of Elements
        # self.z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for _, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
            # get the atom ID number in The Periodic Table of Elements
            self.z.append(atom.GetAtomicNum())

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features  ->  Upper triangular matrix
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                # There judge bond exist
                if bond is None:
                    continue
                f_bond = bond_features(bond=bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # if [a1, a2] in self.reac_idx:
                #     self.bbr.append(self.f_atoms[a1] + f_bond)
                #     self.bbr.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2   a1 is match index of node  == atom -> bond :index
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
                self.bonds.append(np.array([a1, a2]))


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs, args: Namespace):

        if args.gpu:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.args = args
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + self.atom_fdim  # * 2
        self.hedge_bonds_mask = None
        self.hedge_atoms_mask = None
        self.bonds_mask = None
        self.atoms_mask = None

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 0    # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 0    # number of bonds (start at 1 b/c need index 0 as padding)
        self.n_fgs = 0
        self.atom_num = []
        self.fg_num = []
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.fg_scope = []
        self.n_hedge_num = []
        self.frag_scope = []

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms, f_bonds, f_fgs, z, pos, f_fragment, fragment_ap, *graph_data = self.build_data(mol_graphs)
        hedge_group, hedge_atoms_features, hedge_bonds_features, hedge_num, bonds, hedge_indices = graph_data

        # Calculate the max node
        max_node_num = max([atom.shape[0] for atom in f_atoms])
        # max_edge_num = max([bond.shape[0] for bond in f_bonds])

        # self.max_hedge_node_num = max([max([len(group) for group in hedge]) for hedge in hedge_group])   # 5
        self.max_hedge_node_num = 0

        # self.max_hedge_list_num = max([len(hedge) for hedge in hedge_group])     # 67

        temp_hedge = []
        for heg in hedge_group:
            temp_hedge.append(len(heg))
            for group in heg:
                if group is not None:
                    if len(group) > self.max_hedge_node_num:
                        self.max_hedge_node_num = len(group)

        self.max_hedge_list_num = max(temp_hedge) if max(temp_hedge) > max_node_num else max_node_num
        self.max_hedge_bond_num = max([max([len(edge) for edge in graph]) for graph in hedge_bonds_features])

        # B * max_node_num * atom_dim
        # self.f_atoms = torch.cat([pad_2d_unsqueeze(x, max_node_num) for x in f_atoms])
        # atoms_x = torch.zeros((1, self.atom_fdim), dtype=self.f_atoms.dtype)
        # self.atoms_x = torch.cat((atoms_x, torch.cat(f_atoms, dim=0)), dim=0)
        self.f_atoms = torch.cat(f_atoms, dim=0)
        # aggregate the atoms_x

        # B * max_bond_num * bond_dim
        # self.f_bonds = torch.cat([pad_2d_unsqueeze(x, max_edge_num) for x in f_bonds])
        self.f_bonds = torch.cat(f_bonds, dim=0)
        # B * max_hedge_list * max_hedge_node_num
        self.f_fgs = torch.cat(f_fgs, dim=0)      # B * max_fg_num * fg_dim
        # self.z = torch.cat([pad_1d_unsqueeze_nan(torch.Tensor(x), max_node_num) for x in z])    # B * max_node_num
        if self.args.train.use_geometric:
            self.z = z
            self.pos = torch.cat(pos, dim=0)
        else:
            self.z = None
            self.pos = None
        self.fragment_ap = fragment_ap
        self.f_fragment = torch.cat(f_fragment, dim=0)
        # self.reac_idx = reac_idx
        # self.bbr = torch.cat(bbr, dim=0)

        # build the graph
        self.graph_bonds = copy.deepcopy(bonds)

        bonds = np.array(bonds).transpose(1, 0)
        self.bonds = torch.LongTensor(bonds)

        # build the graph and hypergraph
        self.mol_graph = Graph(self.n_atoms, self.graph_bonds)
        self.mol_hyper_graph = Hypergraph.from_graph_kHop(self.mol_graph, 1)

        # self.hedge_atoms_scope = [[len(edge) for edge in graph] for graph in hedge_atoms_features]
        # self.hedge_bonds_scope = [[len(edge) for edge in graph] for graph in hedge_bonds_features]

        f_hedge_atoms_list = []
        hg_hedge_atom_indices = []
        hedge_atoms_indices = []
        last_hedge = 0
        for num_mol, mol_a in enumerate(hedge_atoms_features):
            for num_edge, edge_a in enumerate(mol_a):
                hg_hedge_atom_indices.append(num_mol)
                for node in edge_a:
                    f_hedge_atoms_list.append(node)
                    hedge_atoms_indices.append(num_edge + last_hedge)
            last_hedge += len(mol_a)

        f_hedge_bonds_list = []
        hg_hedge_bonds_indices = []
        hedge_bonds_indices = []
        last_hedge = 0
        for num_mol, mol_b in enumerate(hedge_bonds_features):
            for num_edge, edge_b in enumerate(mol_b):
                hg_hedge_bonds_indices.append(num_mol)
                for e in edge_b:
                    f_hedge_bonds_list.append(e)
                    hedge_bonds_indices.append(num_edge + last_hedge)
            last_hedge += len(mol_b)

        # self.hedge_atoms_pad_features = torch.cat(
        #     [pad_hedge_unsqueeze(x, max_hedge_list_num, max_hedge_node_num, self.atom_fdim)
        #      for x in hedge_atoms_features])
        # self.hedge_bonds_pad_features = torch.cat(
        #     [pad_hedge_unsqueeze(x, max_hedge_list_num, max_hedge_bond_num, self.bond_fdim)
        #      for x in hedge_bonds_features])

        self.hedge_num_scope = [item for sublist in [[i] * item for i, item in enumerate(hedge_num[1])] for item in sublist]
        self.hedge_num = hedge_num

        self.hedge_atoms_features = torch.FloatTensor(f_hedge_atoms_list)
        self.hedge_bonds_features = torch.FloatTensor(f_hedge_bonds_list)
        self.hg_hedge_atoms_indices = torch.tensor(hg_hedge_atom_indices)
        self.hg_hedge_bonds_indices = torch.tensor(hg_hedge_bonds_indices)
        self.hedge_atoms_scope = torch.tensor(hedge_atoms_indices)
        self.hedge_bonds_scope = torch.tensor(hedge_bonds_indices)

        self.hedge_bonds_indices = [item for sublist in hedge_indices for subsublist in sublist for item in subsublist]

    def get_components(self):
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """

        return (self.f_atoms.to(self.device), self.f_bonds.to(self.device), self.f_fgs.to(self.device),
                self.atom_num, self.fg_num)

    # def get_b2b(self) -> torch.LongTensor:
    #     """
    #     Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
    #     :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
    #     """
    #     if self.b2b is None:
    #         b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
    #         # b2b includes reverse edge for each bond so need to mask out
    #         rev_mask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
    #         self.b2b = b2b * rev_mask
    #     return self.b2b

    # def get_a2a(self) -> torch.LongTensor:
    #     """
    #     Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
    #
    #     :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
    #     """
    #     if self.a2a is None:
    #         # b = a1 --> a2
    #         # a2b maps a2 to all incoming bonds b
    #         # b2a maps each bond b to the atom it comes from a1
    #         # thus b2a[a2b] maps atom a2 to neighboring atoms a1
    #         self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds
    #     return self.a2a

    def get_graph(self):
        """
        Get the graph and hyper graph

        :return: Graph, HyperGraph
        """
        return self.mol_graph.to(self.device), self.mol_hyper_graph.to(self.device)

    def get_hyperedge_feature(self):
        """
        Get the number of hyperedge  and the hyperedge group
        The funciton have deprecated.
        :return:
        """
        return (self.hedge_atoms_features.to(self.device), self.hedge_bonds_features.to(self.device),
                self.hg_hedge_atoms_indices.to(self.device), self.hg_hedge_bonds_indices.to(self.device),
                self.hedge_atoms_scope.to(self.device), self.hedge_bonds_scope.to(self.device),
                self.hedge_num_scope, self.hedge_num)

    def get_graph_bonds(self):
        r"""
        Get the bonds and the number of bonds in the graph
        :return:
        """
        return self.graph_bonds, self.hedge_bonds_indices

    # def get_hedge_pad_features(self):
    #     """
    #     Get the pad features of hyperedges
    #     :return:
    #     """
    #     return self.hedge_atoms_pad_features.to(self.device), self.hedge_bonds_pad_features.to(self.device)

    def generate_hedge_features(self, f_atoms, f_bonds, hedge_group, edge_list):
        '''
        padding generate the hedge atoms and bonds features
        :param f_atoms:
        :param f_bonds:
        :param hedge_group:
        :param edge_list:
        :return:
        '''
        hedge_atoms_feature = [[f_atoms[atom] for atom in atoms] for atoms in hedge_group]
        # get the edge pair
        edge_tuple = [list(itertools.permutations(atoms, 2)) for atoms in hedge_group]
        # get the bonds index
        edge_indices = [[edge_list.index(list(pair)) for pair in bond if list(pair) in edge_list] for bond in edge_tuple]
        # get the bonds features
        hedge_bonds_feature = [[f_bonds[index] for index in e_index] for e_index in edge_indices]
        # record the hyperedge indices.
        # edge_indices = [[edge_list[edge] for edge in e_index]for e_index in edge_indices]

        return hedge_atoms_feature, hedge_bonds_feature, edge_indices

    def get_scope(self):
        r"""
        Get the atom bond and fg scope
        :return:
        """
        return (torch.tensor(self.a_scope).to(self.device), torch.tensor(self.b_scope).to(self.device),
                torch.tensor(self.fg_scope).to(self.device), torch.tensor(self.frag_scope).to(self.device))

    def get_geometric_data(self):
        r"""
        Get the graph geometric. contain atom chemistry number, moleculer fragment information and disrupt bonds
        :return:
        """
        if self.args.train.use_geometric:
            return (torch.tensor(self.z).to(self.device), self.pos.to(dtype=torch.float32, device=self.device),
                torch.tensor(self.fragment_ap).to(self.device), self.f_fragment.to(self.device))
        else:
            return self.z, self.pos, torch.tensor(self.fragment_ap).to(self.device), self.f_fragment.to(self.device)

    # def get_mask(self):
    #     # self.atoms_mask = generate_mask(self.atom_num, self.f_atoms.shape, '2')
    #     # self.bonds_mask = generate_mask(self.bonds_num, self.f_bonds.shape, '2')
    #
    #     self.hedge_atoms_mask = generate_mask(self.hedge_atoms_scope, self.hedge_atoms_features.shape, '3')
    #     self.hedge_bonds_mask = generate_mask(self.hedge_bonds_scope, self.hedge_bonds_features.shape, '3')
    #     return (self.hedge_atoms_mask.to(self.device), self.hedge_bonds_mask.to(self.device))

    def build_data(self, mol_graphs):
        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = []
        f_bonds = []
        f_fgs = []  # fg features
        # a2b = []  # mapping from atom index to incoming bond indices
        b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond
        # bonds = [[0, 0]]
        bonds = []
        z = []  # get the atom ID number in The Periodic Table of Elements
        pos = []
        fragment_ap = []
        f_fragment = []
        hedge_bonds_indices = []

        hedge_group = []
        hedge_atoms_features = []
        hedge_bonds_features = []
        hedge_num = []
        hedge_num_org = []
        last_num_bonds = 0
        frag_num = 0
        frag_a_scope = []
        index = 0
        for i, mol_graph in enumerate(mol_graphs):
            edge_list = []          # for each molecule graph
            # a2b: atom have bonds.
            # for a in range(mol_graph.n_atoms):
            #     a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])        # a2b:1:[2, 3, 4], 2:[1, 3, 6], 3:[1, 2]

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
                edge_list.append([mol_graph.b2a[b], mol_graph.b2a[mol_graph.b2revb[b]]])            # it can deprecated.

            # check the edge list
            assert len(mol_graph.f_bonds) == len(edge_list), \
                (f"the f_bonds len is {len(mol_graph.f_bonds)}, the bond len is {len(mol_graph.bonds)} "
                 f"the n_bonds is {mol_graph.n_bonds} the bonds len is {len(edge_list)}  {edge_list}")

            self.a_scope.extend([i] * mol_graph.n_atoms)
            self.b_scope.extend([i] * mol_graph.n_bonds)
            self.fg_scope.extend([i] * mol_graph.n_fgs)
            self.atom_num.append(mol_graph.n_atoms)
            self.fg_num.append(mol_graph.n_fgs)
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
            self.n_fgs += mol_graph.n_fgs

            # build Graph and HyperGraph
            graph = Graph(mol_graph.n_atoms, mol_graph.bonds)

            # hypergraph = Hypergraph.from_graph(graph)         # deprecated.
            # Build the hypergraph of k-hop
            hypergraph = Hypergraph.from_graph_kHop(graph, 1)
            hedge_num_org.append(hypergraph.num_e)

            # build the hyper edge, delete the isolated atoms
            hedge = [i for i in hypergraph.e[0] if len(i) > 1]
            hedge_group.append(hedge)
            hedge_num.append(len(hedge))

            # build the hyper atom and bonds feature
            hedge_atom_features, hedge_bond_features, hedge_indices = self.generate_hedge_features(mol_graph.f_atoms, mol_graph.f_bonds, hedge, edge_list)

            if len(hedge_atom_features) < hypergraph.num_e:
                temp_a = [0] * get_atom_fdim()
                temp_b = [0] * (get_bond_fdim() + get_atom_fdim())
                temp_a_list = []
                temp_b_list = []

                temp_a_list.append(temp_a)
                temp_a_list.append(temp_a)

                temp_b_list.append(temp_b)
                temp_b_list.append(temp_b)

                for i in range(hypergraph.num_e - len(hedge_atom_features)):
                    hedge_atom_features.append(temp_a_list)
                    hedge_bond_features.append(temp_b_list)

            hedge_atoms_features.append(hedge_atom_features)
            hedge_bonds_features.append(hedge_bond_features)
            hedge_indices = [[bonds[item+last_num_bonds] for item in indices] for indices in hedge_indices]
            hedge_bonds_indices.append(hedge_indices)

            last_num_bonds = len(bonds)
            self.frag_scope.extend([index] * len(mol_graph.f_fragment))

            if self.args.train.use_geometric:
                z.extend(mol_graph.z)
                pos.append(torch.tensor(mol_graph.coords))

            fragment_ap.append(mol_graph.fragment_ap)
            f_fragment.append(torch.tensor(mol_graph.f_fragment))

            # reac_idx.append(mol_graph.reac_idx)
            # bbr.append(torch.tensor(mol_graph.bbr))

            f_atoms.append(torch.FloatTensor(mol_graph.f_atoms))
            f_bonds.append(torch.FloatTensor(mol_graph.f_bonds))
            f_fgs.append(torch.FloatTensor(mol_graph.f_fgs))

            mol_graph.fragment_ap.sort(key=lambda x: x[0])
            for k, v in mol_graph.fragment_ap:
                frag_a_scope.extend([v + frag_num])

            frag_num += max([v for k,v in mol_graph.fragment_ap]) + 1
            index += 1

        index = 0
        return (f_atoms, f_bonds, f_fgs, z, pos, f_fragment, frag_a_scope,  hedge_group,
                hedge_atoms_features, hedge_bonds_features, [hedge_num_org, hedge_num], bonds, hedge_bonds_indices)


def mol2graph(smiles_batch: List[str], mols_batch: List, args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param mols_batch:
    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = [MolGraph(args, smiles_batch[i], mols_batch[i]) for i in range(len(smiles_batch))]

    return BatchMolGraph(mol_graphs, args)


def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")
    - .npz (assumes features are saved with name "features")
    - .npy
    - .csv/.txt (assumes comma-separated features with a header and with one line per molecule)
    - .pkl/.pckl/.pickle containing a sparse numpy array (TODO: remove this option once we are no longer dependent on it)

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def generate_mask(data, shape, type='2'):
    mask_shape = shape[:-1]
    new_x_mask = torch.zeros(mask_shape, dtype=torch.float)
    for i in range(len(data)):
        for j in range(len(data[i])):
            new_x_mask[i, j, :data[i][j]] = 1
    # if type == '2':
    #     for l in range(len(data)):
    #         new_x_mask[l, :data[l]] = 1
    # else:
    #     for i in range(len(data)):
    #         for j in range(len(data[i])):
    #             new_x_mask[i, j, :data[i][j]] = 1

    return new_x_mask


def mol2centroid(mol):
    r"""
    Get the centroid of molecule
    :param mol:
    :return:
    """
    lig_xyz = mol.GetConformer().GetPositions()
    centroid_x, centroid_y, centroid_z = lig_xyz.mean(axis=0)
    return centroid_x, centroid_y, centroid_z
