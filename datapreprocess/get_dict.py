
from rdflib import Graph, URIRef
from rdkit import Chem
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from typing import Tuple
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# VSCode and PyCharm Path have different.
# VSCode can use cwd, The PyCharm use parent
cur_path = Path.cwd()
parent_path = cur_path.parent


def get_onto_emb(filename):
    """
    get the all embedding of element and functional groups 
    """
    return KeyedVectors.load_word2vec_format(filename, binary=False)


def get_fg_name(filename) -> list:
    """
    get all functional groups name
    """
    with open(filename, 'r') as f:
        func_groups = f.read().strip().split('\n')
        name = [i.split()[0] for i in func_groups]
    
    return name


# get property -> embedding matrix
def get_property(filename) -> list:
    """
    # get property -> embedding matrix
    """
    obj_property_list = []
    with(open(filename, 'r')) as f:
        for line in f.readlines():
            temp = line.split()
            obj_property_list.extend(temp)

    return obj_property_list


def get_emb_dict(emb_data, element, emb_files, save_flag=False) -> dict:
    """
    get embedding dict
    """
    emb_dict = {}
    for item in element:
        fg_name = "http://www.semanticweb.org/ElementKG#" + item
        ele_emb = emb_data[fg_name]
        emb_dict[item] = ele_emb
    
    if save_flag:
        #  functional groups embedding
        pickle.dump(emb_dict, open(emb_files,'wb'))
    
    return emb_dict


def get_property_emb(emb_data, property) -> Tuple[list, list]:
    """
    get valid property and embedding
    """
    valid_pro = []
    pro_emb = []
    for p in property:
        p_name = "http://www.semanticweb.org/ElementKG#" + p
        if p_name in emb_data:
            valid_pro.append(p)
            pro_emb.append(emb_data[p_name])
    
    return (valid_pro, pro_emb)


# get all element symbol
def get_atom_symbol(atomic_number):
    '''
    get all element symbol
    '''
    return Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), atomic_number)


# get property -> embedding dict
def get_property_utlimate_emb_dict(org_emb_data, pro_list) -> dict:
    """
    get new propery embedding dict
    """
    property_emb = {}
    for p in range(len(pro_list)):
        p_name = URIRef(f"http://www.semanticweb.org/ElementKG#" + pro_list[p])
        property_emb[p_name] = org_emb_data[p]
    
    return property_emb


def get_relation_emb(pro_emb, ele_symbols, g_files, save_file, save_flag = False) -> dict:
    """
    get relation embedding
    """
    g = Graph()
    g.parse(g_files, format="xml")

    print("getting rel2emb dict ...")
    rel_emb = {}
    for i in range(len(ele_symbols) - 1):
        for j in range(1, len(ele_symbols)):
            #  SparQL
            #  query the relation
            qr = "select ?relation where { \
                 <http://www.semanticweb.org/ElementKG#" + ele_symbols[i] + "> \
                  ?relation <http://www.semanticweb.org/ElementKG#"+ele_symbols[j]+">}"

            relations = g.query(qr)
            relations = list(relations)
            relations = [pro_emb[rel[0]] for rel in relations]
            
            if relations:
                relation = np.mean(relations, axis=0)
                rel_emb[(i, j)] = relation
                rel_emb[(j, i)] = relation
    
    if save_flag:
        pickle.dump(rel_emb, open(save_file, 'wb'))

    return rel_emb


def main():
    # print(cur_path)
    # VSCode
    # data_path = cur_path.joinpath('data')
    # print(parent_path)
    # PyCharm
    data_path = parent_path.joinpath('data')
    onto_path = data_path.joinpath('elementkgontology.embeddings.txt')
    onto_emb = get_onto_emb(onto_path)
    fg_path = data_path.joinpath('funcgroup.txt')
    fg_names = get_fg_name(fg_path)
    
    # save flag
    flag = False

    print('getting fg2emb dict ...')
    fg_emb = get_emb_dict(onto_emb, fg_names, data_path.joinpath('fg2emb.pkl'), flag)
    element_symbols = [get_atom_symbol(i) for i in range(1, 109)]

    print("getting ele2emb dict ...")
    ele_emb = get_emb_dict(onto_emb, element_symbols, data_path.joinpath('ele2emb.pkl'), flag)

    obj_pro_path = data_path.joinpath('objectproperty.txt')
    obj_property_matrix = get_property(obj_pro_path)    # len:  93

    v_property, pro_emb = get_property_emb(onto_emb, obj_property_matrix)
    
    # list convert matrix
    pro_emb_matrix = np.concatenate(pro_emb, axis=0)
    
    # PCA
    pca = PCA(n_components=14)
    pro_emb = pca.fit_transform(pro_emb)    # shape: len * dim | 93 * 14

    print("getting property2emb dict ...")
    property_emb_dict = get_property_utlimate_emb_dict(pro_emb, v_property)
    # print(property_emb_dict)
    graph_path = data_path.joinpath('elementkg.owl')
    rel_emb = get_relation_emb(property_emb_dict, element_symbols,graph_path, data_path.joinpath('rel2emb.pkl'), flag)
    
    print("finish!")


if __name__ == '__main__':
    main()