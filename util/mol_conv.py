import pandas
import torch
import dgl
import numpy as np
# 计算分子描述符
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
# # mendeleev为0.5.1
from mendeleev import get_table
from sklearn.utils import shuffle


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sel_prop_names = ['atomic_weight',
                'atomic_radius',
                'atomic_volume',
                'dipole_polarizability',
                'fusion_heat',
                'thermal_conductivity',
                'vdw_radius',
                'en_pauling']
dim_atomic_feat = len(sel_prop_names)
dim_self_feat = 3


class molDGLGraph(dgl.DGLGraph):
    def __init__(self, smiles, adj_mat, feat_mat, mol):
        super(molDGLGraph, self).__init__()
        self.smiles = smiles
        self.adj_mat = adj_mat
        self.feat_mat = feat_mat
        # 原子符号
        self.atomic_nodes = []
        # {原子符号：原子符号的两个邻接原子}
        self.neighbors = {}

        node_id = 0
        for atom in mol.GetAtoms():
            self.atomic_nodes.append(atom.GetSymbol())
            self.neighbors[node_id] = atoms_to_symbols(atom.GetNeighbors())
            node_id += 1


def read_atom_prop():
    '''
    元素周期表共118个元素，此函式返回{元素周期表：对应元素的8种性质}字典集合
    8种性质分别为：atomic_weight、atomic_radius、atomic_volume、dipole_polarizability、
                 fusion_heat、thermal_conductivity、vdw_radius、en_pauling
    :return:
    '''
    tb_atomic_props = get_table('elements')  # type: # DataFrame
    arr_atomic_nums = np.array(tb_atomic_props['atomic_number'], dtype=np.int)  # (118,)
    # np.nan_to_num: 用零替换NaN，用最大的有限数替换无穷大
    arr_atomic_props = np.nan_to_num(np.array(tb_atomic_props[sel_prop_names], dtype=np.float32)) # shape: (118,8)
    # util.zscore 标准化
    arr_atomic_props = util.zscore(arr_atomic_props)  # shape: (118,8)
    # 字典格式，将原子序号和原子信息一一对应
    atomic_props_mat = {arr_atomic_nums[i]: arr_atomic_props[i, :] for i in range(0, arr_atomic_nums.shape[0])}

    return atomic_props_mat


def construct_mol_graph(smiles, mol, adj_mat, feat_mat):
    # feat_mat = node_feat_mat
    molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol)
    edges = util.adj_mat_to_edges(adj_mat)
    # zip函数：[(1,2),(3,4)] --->  [(1,3),(2,4)]
    src, dst = tuple(zip(*edges))

    molGraph.add_nodes(adj_mat.shape[0])   # 添加节点
    molGraph.add_edges(src, dst)   # 添加边
    # molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32).to(device)
    molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32)
    return molGraph


def smiles_to_mol_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)   # 生成邻接矩阵 shanpe：原子数*原子数
        # node_feat_mat: 原子数 * 原子的8种性质
        node_feat_mat = np.empty([mol.GetNumAtoms(), atomic_props.get(1).shape[0]])
        ind = 0
        for atom in mol.GetAtoms():
            # 原子元素周期编号: atom.GetAtomicNum()
            # 字典取值：dict.get(key) = value
            node_feat_mat[ind, :] = atomic_props.get(atom.GetAtomicNum())
            ind = ind + 1

        return mol, construct_mol_graph(smiles, mol, adj_mat, node_feat_mat)
    except:
        print(smiles + ' could not be converted to molecular graph due to the internal errors of RDKit')
        return None, None


def atoms_to_symbols(atoms):
    symbols = []

    for atom in atoms:
        symbols.append(atom.GetSymbol())

    return symbols


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


def read_dataset(file_name):
    samples = []
    mol_graphs = []
    targets = []
    data_mat = np.array(shuffle(pandas.read_csv(file_name)))
    smiles = data_mat[:, 0]                              # shape: (1128,)
    target = np.array(data_mat[:, -1])  # shape: (1128, 1)

    for i in range(0, data_mat.shape[0]):  # 遍历每一个分子
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()   # 原子数目
            mol_graph.weight = dsc.ExactMolWt(mol)   # 分子重量
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

    return samples


atomic_props = read_atom_prop()


# from rdkit.Chem import Draw
# mol = Chem.MolFromSmiles('c1ccccc1C(=O)O')
# Draw.MolToImage(mol)

# # 计算分子性质
# m = Chem.MolFromSmiles('c1ccccc1C(=O)O')
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors
# tpsa_m = Descriptors.TPSA(m)  # 计算分子的The topological polar surface area (TPSA) descriptor 、
# logp_m = Descriptors.MolLogP(m)  # 计算logp
# mw = Descriptors.ExactMolWt(m)   # 计算分子重量
# AllChem.ComputeGasteigerCharges(m)  # 计算电荷