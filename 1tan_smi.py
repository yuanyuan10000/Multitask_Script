import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as dist  # 导入scipy距离公式
from sklearn.utils import shuffle
import pandas as pd
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols


# def tanimoto_coefficient(p_vec, q_vec):
#     """
#     This method implements the cosine tanimoto coefficient metric
#     :param p_vec: vector one
#     :param q_vec: vector two
#     :return: the tanimoto coefficient between vector one and two
#     """
#     pq = np.dot(p_vec, q_vec)
#     p_square = np.linalg.norm(p_vec)
#     q_square = np.linalg.norm(q_vec)
#     return pq / (p_square + q_square - pq)


# def correlation(df1,df2):
#     tani_coef = []
#     avg_tani_coef = []
#     for j in range(df1.shape[0]):
#         for i in range(df2.shape[0]):
#             b = dist.pdist(pd.concat([df1.iloc[j,:],df2.iloc[i,:]],axis=1).T, 'jaccard')   # 0.8571
#             tani_coef.append(b.item())
#         avg_tani_coef.append(np.array(tani_coef).mean())
#     return np.array(avg_tani_coef).mean()


# def tanimoto_simlarity(morgan_fps1,morgan_fps2):
#     mean_dist_list = []
#     for i in range(len(morgan_fps1)):
#         dist = np.array(DataStructs.BulkTanimotoSimilarity(morgan_fps1[i], morgan_fps2, returnDistance=True))
#         mean_dist = np.mean(np.array(dist))
#         mean_dist_list.append(mean_dist)
#     return np.array(mean_dist_list).mean()

# def tanimoto_simlarity(morgan_fps1,morgan_fps2):
#     dist = 1 - np.array([np.mean(DataStructs.BulkTanimotoSimilarity(f, morgan_fps2)) for f in morgan_fps1])
#     return np.array(dist).mean()


def tanimoto_simlarity(morgan_fps1,morgan_fps2):
    dist = []
    for i in range(len(morgan_fps1)):
        similarity = np.array([DataStructs.FingerprintSimilarity(morgan_fps1[i], f) for f in morgan_fps2]).max()
        dist.append(similarity)
    return np.array(dist).mean()

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

save_path = "I:/4Multitask/data/"
data_path = "I:/4Multitask/exp/Descriptors/train"

# tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
#              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
             'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

data_listname = []
data_listname = sorted(listdir(data_path, data_listname))

for i in range(12):
    locals()['df'+str(i)] = np.ravel(pd.read_csv(data_listname[i],usecols=['smiles']))
    locals()['mols'+str(i)] = [Chem.MolFromSmiles(m) for m in locals()['df'+str(i)]]
    locals()['morgan_fps'+str(i)] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in locals()['mols'+str(i)]]


tan_smi = np.empty((12,12),dtype=np.float32)
for m in range(12):
    print('*'*12+'正在处理第{}行数据'.format(m)+'*'*12)
    for n in range(12):
        if m == n:
            tan_smi[m, n] = 1
        else:
            tan_smi[m,n] = tanimoto_simlarity(locals()['morgan_fps'+str(m)],locals()['morgan_fps'+str(n)])
        print('similarity: {}'.format(tan_smi[m,n]))

df_smi = pd.DataFrame(tan_smi)
df_smi.columns = tasks_name
df_smi.index = tasks_name
df_smi.to_csv('data_FingerprintSimilarity.csv')



################################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

collect_nbrs = pd.read_csv('./data/data_FingerprintSimilarity.csv')
collect_nbrs = collect_nbrs.iloc[:,1:]

f, ax = plt.subplots(figsize = (10,8))
# plt.title('Similarity Matrix of SYK',fontsize=15,fontweight='bold')
h = sns.heatmap(collect_nbrs,ax=ax,vmax=1.0, vmin=0.7, cmap='GnBu',
            yticklabels=collect_nbrs.columns.to_list(),xticklabels=collect_nbrs.columns.to_list(),
            # annot=False,  # 显示数值
            # linewidths=.5,  # 设置每个单元格边框的宽度
            cbar=False # 不显示颜色刻度条
             )
#设置坐标字体方向
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# 设置颜色条刻度字体的大小
cb = h.figure.colorbar(h.collections[0],drawedges=False) #显示colorbar
cb.ax.tick_params(labelsize=20)  # 设置colorbar刻度字体大小。

# plt.title('Tanimoto similarity between the different assays of Tox21',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('Tanimoto similarity between the different assays of Tox21.png')
plt.show()


# #
# df1 = np.array(pd.read_csv(data_listname[0],usecols=['smiles'])).reshape(-1)
# df2 = np.array(pd.read_csv(data_listname[1],usecols=['smiles'])).reshape(-1)
# mols1 = [Chem.MolFromSmiles(m) for m in df1]
# print(len(mols1))
# mols2 = [Chem.MolFromSmiles(m) for m in df2]
# print(len(mols2))
#
#
# # # MACCS Keys
# # maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mols]
# # maccs = DataStructs.BulkTanimotoSimilarity(maccs_fps[0], maccs_fps[1:])
# # Rdkit
# # rdkit_fps = [Chem.Fingerprints.FingerprintMols.FingerprintMol(mol) for mol in mols]
# # rdkit = DataStructs.BulkTanimotoSimilarity(rdkit_fps[0], rdkit_fps[1:])
#
# morgan_fps1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols1]
# morgan_fps2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols2]
#
# mean_dist = []
# for i in range(len(mols1)):
#     dist = np.array(DataStructs.BulkTanimotoSimilarity(morgan_fps1[i], morgan_fps2,returnDistance=True))
#     mean_dist.append(np.mean(dist))
# np.array(mean_dist).mean()




# from numpy import *
# import scipy.spatial.distance as dist  # 导入scipy距离公式
# matV = mat([[1,1,0,1,0,1,0,0,1],[0,1,1,0,0,0,1,1,1],[1,1,0,0,1,0,0,0,1],[0,1,0,0,1,0,0,1,0]])
# print ("dist.jaccard:", dist.pdist(matV,'jaccard'))
#
# def correlation(set_a,set_b):
#     unions = len(set_a.union(set_b))
#     intersections = len(set_a.intersection(set_b))
#     return 1. * intersections / unions


