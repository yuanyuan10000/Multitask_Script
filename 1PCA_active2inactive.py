import pandas as pd
# 计算分子描述符
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
from dgllife.utils import *
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from rdkit.Chem import Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

def cal_desc(smile):
    ##### rdkit.Chem.Lipinski模块
    # 氢键受体数NumHAcceptors
    # 氢键供体数NumHDonors
    # 可旋转键数NumRotatableBonds
    # 脂肪环数量NumAliphaticRings
    # 芳香环数量NumAromaticRings
    # SP3杂化碳原子比例FractionCSP3

    #### rdkit.Chem.Descriptors模块
    # 拓扑极表面积TPSA
    # 脂水分配系数MolLogP
    # 分子量MolWt
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)  # 加氢
    # rdkit.ML.Descriptors.MoleculeDescriptors模块
    des_list = ['NumHAcceptors','NumHDonors','NumRotatableBonds','NumAliphaticRings','NumAromaticRings','TPSA', 'MolWt']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    # 查看各种描述符的含义：GetDescriptorSummaries()
    # des_list = calculator.GetDescriptorSummaries()
    # 获取所有描述符：Descriptors._descList
    # des_list = [x[0] for x in Descriptors._descList]
    # len(des_list)
    mol_property = [i for i in calculator.CalcDescriptors(mol)]

    mol_property.extend([mol.GetNumAtoms()])
    des_list.extend(['NumAtoms'])
    return mol_property,des_list

def split_smi(path,frac):
    print('正在处理文件{}'.format(os.path.basename(path)))
    df = shuffle(pd.read_csv(path), random_state=23)
    df_sample = df.sample(frac=frac, replace=False, random_state=18)  # replace：是否为有放回抽样，取replace=True时为有放回抽样。
    df_NonTox = df[df_sample.iloc[:, -1] == 0].reset_index().drop(columns=['index'])
    df_Tox = df[df_sample.iloc[:, -1] == 1].reset_index().drop(columns=['index'])
    smi_Tox = df_Tox.iloc[:,0].ravel()
    smi_NonTox = df_NonTox.iloc[:,0].ravel()
    return smi_Tox,smi_NonTox


def cal_tsne(smiles,n_components):
    #  生成特征
    df_desc = []
    for smi in smiles:
        mol_property,des_list = cal_desc(smi)
        df_desc.append(mol_property)
    df_desc = pd.DataFrame(df_desc,columns=des_list)
    # 降维
    transfer = PCA(n_components=n_components)
    df_pca = transfer.fit_transform(df_desc)
    return df_pca, df_desc


save_path = "I:/4Multitask/exp/"
data_path = 'I:/4Multitask/data/data'



tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

data_listname = []
data_listname = sorted(listdir(data_path, data_listname))



pca_merge_tox, pca_merge_nontox = [],[]
for i in range(12):
    smi_tox,smi_nontox = split_smi(path=data_listname[i], frac=1)
    df_pca_tox, df_desc_tox = cal_tsne(smi_tox,n_components=2)
    df_pca_nontox, df_desc_nontox = cal_tsne(smi_nontox,n_components=2)
    pca_merge_nontox.append(df_pca_nontox)
    pca_merge_tox.append(df_pca_tox)






leg_fs = 17
tit_ls = 20
size = 12
label1 = 'Toxicity'
label2 = 'Nontoxicity'

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15,10), dpi=100)

axes[0,0].scatter(pca_merge_nontox[0][:,0], pca_merge_nontox[0][:,1], s = size,color = 'cornflowerblue',label=label2)
axes[0,0].scatter(pca_merge_tox[0][:,0], pca_merge_tox[0][:,1], s = size, color = 'darkorange', label=label1)
axes[0,0].set_title(tasks_name[0],fontsize=tit_ls)
axes[0,0].legend(loc='upper right', fontsize=leg_fs)

axes[0,1].scatter(pca_merge_nontox[1][:,0], pca_merge_nontox[1][:,1], s = size, color = 'cornflowerblue',label=label2)
axes[0,1].scatter(pca_merge_tox[1][:,0], pca_merge_tox[1][:,1], s = size, color = 'darkorange', label=label1)
axes[0,1].set_title(tasks_name[1],fontsize=tit_ls)
axes[0,1].legend(loc='upper right', fontsize=leg_fs)

axes[0,2].scatter(pca_merge_nontox[2][:,0], pca_merge_nontox[2][:,1], s = size,color = 'cornflowerblue',label=label2)
axes[0,2].scatter(pca_merge_tox[2][:,0], pca_merge_tox[2][:,1], s = size, color = 'darkorange', label=label1)
axes[0,2].set_title(tasks_name[2],fontsize=tit_ls)
axes[0,2].legend(loc='upper right', fontsize=leg_fs)

axes[0,3].scatter(pca_merge_nontox[3][:,0], pca_merge_nontox[3][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[0,3].scatter(pca_merge_tox[3][:,0], pca_merge_tox[3][:,1], s = size, color = 'darkorange', label=label1)
axes[0,3].set_title(tasks_name[3],fontsize=tit_ls)
axes[0,3].legend(loc='upper right', fontsize=leg_fs)

axes[1,0].scatter(pca_merge_nontox[4][:,0], pca_merge_nontox[4][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[1,0].scatter(pca_merge_tox[4][:,0], pca_merge_tox[4][:,1], s = size, color = 'darkorange', label=label1)
axes[1,0].set_title(tasks_name[4],fontsize=tit_ls)
axes[1,0].legend(loc='upper right', fontsize=leg_fs)

axes[1,1].scatter(pca_merge_nontox[5][:,0], pca_merge_nontox[5][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[1,1].scatter(pca_merge_tox[5][:,0], pca_merge_tox[5][:,1], s = size, color = 'darkorange', label=label1)
axes[1,1].set_title(tasks_name[5],fontsize=tit_ls)
axes[1,1].legend(loc='upper right', fontsize=leg_fs)

axes[1,2].scatter(pca_merge_nontox[6][:,0], pca_merge_nontox[6][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[1,2].scatter(pca_merge_tox[6][:,0], pca_merge_tox[6][:,1], s = size, color = 'darkorange', label=label1)
axes[1,2].set_title(tasks_name[6],fontsize=tit_ls)
axes[1,2].legend(loc='upper right', fontsize=leg_fs)

axes[1,3].scatter(pca_merge_nontox[7][:,0], pca_merge_nontox[7][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[1,3].scatter(pca_merge_tox[7][:,0], pca_merge_tox[7][:,1], s = size, color = 'darkorange', label=label1)
axes[1,3].set_title(tasks_name[7],fontsize=tit_ls)
axes[1,3].legend(loc='upper right', fontsize=leg_fs)

axes[2,0].scatter(pca_merge_nontox[8][:,0], pca_merge_nontox[8][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[2,0].scatter(pca_merge_tox[8][:,0], pca_merge_tox[8][:,1], s = size, color = 'darkorange', label=label1)
axes[2,0].set_title(tasks_name[8],fontsize=tit_ls)
axes[2,0].legend(loc='upper right', fontsize=leg_fs)

axes[2,1].scatter(pca_merge_nontox[9][:,0], pca_merge_nontox[9][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[2,1].scatter(pca_merge_tox[9][:,0], pca_merge_tox[9][:,1], s = size, color = 'darkorange', label=label1)
axes[2,1].set_title(tasks_name[9],fontsize=tit_ls)
axes[2,1].legend(loc='upper right', fontsize=leg_fs)

axes[2,2].scatter(pca_merge_nontox[10][:,0], pca_merge_nontox[10][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[2,2].scatter(pca_merge_tox[10][:,0], pca_merge_tox[10][:,1], s = size, color = 'darkorange', label=label1)
axes[2,2].set_title(tasks_name[10],fontsize=tit_ls)
axes[2,2].legend(loc='upper right', fontsize=leg_fs)

axes[2,3].scatter(pca_merge_nontox[11][:,0], pca_merge_nontox[11][:,1], s = size, color = 'cornflowerblue', label=label2)
axes[2,3].scatter(pca_merge_tox[11][:,0], pca_merge_tox[11][:,1], s = size, color = 'darkorange', label=label1)
axes[2,3].set_title(tasks_name[11],fontsize=tit_ls)
axes[2,3].legend(loc='upper right', fontsize=leg_fs)

plt.tight_layout()

plt.savefig('./exp/Distribution between Tox and NonTxo.png',dim=300)
plt.show()












