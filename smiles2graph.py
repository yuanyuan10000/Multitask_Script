import pandas as pd
import torch
import dgl
import numpy as np
# 计算分子描述符
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
# # mendeleev为0.5.1
from mendeleev import get_table
from rdkit.Chem import AllChem
from dgllife.utils import *
import rdkit
from rdkit.Chem import QED


# 标准化函数 x-mean/std
def normalize_self_feat(mol_graphs, self_feat_name):
    self_feats = []

    for mol_graph in mol_graphs:
        self_feats.append(getattr(mol_graph, self_feat_name))

    mean_self_feat = np.mean(self_feats)
    std_self_feat = np.std(self_feats)

    for mol_graph in mol_graphs:
        if std_self_feat == 0:
            setattr(mol_graph, self_feat_name, 0)
        else:
            setattr(mol_graph, self_feat_name, (getattr(mol_graph, self_feat_name) - mean_self_feat) / std_self_feat)

def bond_featurizer(mol, add_self_loop=False):
    feats = []
    num_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())
    distance_matrix = Chem.GetDistanceMatrix(mol)  # 混淆矩阵
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                feats.append(float(distance_matrix[i, j]))
    return {'edge': torch.tensor(feats).reshape(-1, 1).float()}


def complete_graph(file_name):
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]  # shape: (1128,)
    target = np.array(data_mat[:, -1])

    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    # edge_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    n_feats = node_featurizer.feat_size('feat')  # 节点特征size

    samples = []
    mol_graphs = []
    targets = []
    for i in range(data_mat.shape[0]):
        mol = Chem.MolFromSmiles(smiles[i])
        mol_graph = mol_to_complete_graph(mol, node_featurizer=node_featurizer,edge_featurizer=bond_featurizer)
        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()  # 原子数目
            mol_graph.weight = dsc.ExactMolWt(mol)  # 分子重量
            mol_graph.num_rings = mol.GetRingInfo().NumRings()  # 分子环数目


            # 加入分子电荷信息
            mol_graph.max_abs_charge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.min_abs_charge = dsc.MinAbsPartialCharge(mol)
            mol_graph.num_rad_elc = dsc.NumValenceElectrons(mol)
            mol_graph.num_val_elc = dsc.NumValenceElectrons(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            targets.append(target[i])

    normalize_self_feat(mol_graphs, 'num_atoms')
    normalize_self_feat(mol_graphs, 'weight')
    normalize_self_feat(mol_graphs, 'num_rings')
    normalize_self_feat(mol_graphs, 'max_abs_charge')
    normalize_self_feat(mol_graphs, 'min_abs_charge')
    normalize_self_feat(mol_graphs, 'num_rad_elc')
    normalize_self_feat(mol_graphs, 'num_val_elc')

    return samples, n_feats, np.array(smiles)


def bigraph(file_name):
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]  # shape: (1128,)
    target = np.array(data_mat[:, -1])

    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='edge')
    n_feats = node_featurizer.feat_size('feat')  # 节点特征size

    samples = []
    mol_graphs = []
    targets = []
    for i in range(data_mat.shape[0]):
        mol = Chem.MolFromSmiles(smiles[i])
        mol_graph = mol_to_bigraph(mol, node_featurizer=node_featurizer,edge_featurizer=edge_featurizer)
        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()  # 原子数目
            mol_graph.weight = dsc.ExactMolWt(mol)  # 分子重量
            mol_graph.num_rings = mol.GetRingInfo().NumRings()  # 分子环数目

            mol_graph.mol_logP = dsc.MolLogP(mol)  # 脂水分配系数MolLogP
            mol_graph.NHA = rdkit.Chem.Lipinski.NumHAcceptors(mol)  # 氢键受体
            mol_graph.NHD = rdkit.Chem.Lipinski.NumHDonors(mol)   # 氢键供体
            mol_graph.num_rotatable_bonds = rdkit.Chem.Lipinski.NumRotatableBonds(mol)  # 可旋转键数目
            mol_graph.qed_score = QED.qed(mol)  # 类药性打分


            # 加入分子电荷信息
            mol_graph.max_abs_charge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.min_abs_charge = dsc.MinAbsPartialCharge(mol)
            mol_graph.num_rad_elc = dsc.NumValenceElectrons(mol)
            mol_graph.num_val_elc = dsc.NumValenceElectrons(mol)

            #
            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            targets.append(target[i])

    # 分子特征标准化
    normalize_self_feat(mol_graphs, 'num_atoms')
    normalize_self_feat(mol_graphs, 'weight')
    normalize_self_feat(mol_graphs, 'num_rings')

    normalize_self_feat(mol_graphs, 'mol_logP')
    normalize_self_feat(mol_graphs, 'NHA')
    normalize_self_feat(mol_graphs, 'NHD')
    normalize_self_feat(mol_graphs, 'num_rotatable_bonds')
    normalize_self_feat(mol_graphs, 'qed_score')

    normalize_self_feat(mol_graphs, 'max_abs_charge')
    normalize_self_feat(mol_graphs, 'min_abs_charge')
    normalize_self_feat(mol_graphs, 'num_rad_elc')
    normalize_self_feat(mol_graphs, 'num_val_elc')

    return samples, n_feats, np.array(smiles)


def nearest_neighbor_graph(file_name):
    data_mat = np.array(pd.read_csv(file_name))
    smiles = data_mat[:, 0]  # shape: (1128,)
    target = np.array(data_mat[:, -1])

    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    n_feats = node_featurizer.feat_size('feat')  # 节点特征size

    samples = []
    mol_graphs = []
    targets = []
    for i in range(data_mat.shape[0]):
        mol = Chem.MolFromSmiles(smiles[i])
        AllChem.EmbedMolecule(mol)  # 将二维分子图转化为三维分子坐标
        # AllChem.MMFFOptimizeMolecule(mol)  # 对分子结构进行简单优化
        coords = get_mol_3d_coordinates(mol)
        try:
            mol_graph = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25, keep_dists=True,
                                                    node_featurizer=node_featurizer)
        except:
            print(i+1,smiles[i])
        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()  # 原子数目
            mol_graph.weight = dsc.ExactMolWt(mol)  # 分子重量
            mol_graph.num_rings = mol.GetRingInfo().NumRings()  # 分子环数目

          # 加入分子电荷信息
            mol_graph.max_abs_charge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.min_abs_charge = dsc.MinAbsPartialCharge(mol)
            mol_graph.num_rad_elc = dsc.NumValenceElectrons(mol)
            mol_graph.num_val_elc = dsc.NumValenceElectrons(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)
            targets.append(target[i])

    normalize_self_feat(mol_graphs, 'num_atoms')
    normalize_self_feat(mol_graphs, 'weight')
    normalize_self_feat(mol_graphs, 'num_rings')
    normalize_self_feat(mol_graphs, 'max_abs_charge')
    normalize_self_feat(mol_graphs, 'min_abs_charge')
    normalize_self_feat(mol_graphs, 'num_rad_elc')
    normalize_self_feat(mol_graphs, 'num_val_elc')

    return samples, n_feats, np.array(smiles)