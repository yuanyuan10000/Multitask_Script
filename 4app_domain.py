import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as dist  # 导入scipy距离公式
from sklearn.utils import shuffle
import pandas as pd
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


# 设置active和inactive分子不同形状
def markset(sc,labels):
    map_marker = {0: 'o', 1: 's'}
    m = list(map(lambda x: map_marker[x], labels))
    paths = []
    for marker in m:
        if isinstance(marker, mmarkers.MarkerStyle):
            marker_obj = marker
        else:
            marker_obj = mmarkers.MarkerStyle(marker)
        path = marker_obj.get_path().transformed(
            marker_obj.get_transform())
        paths.append(path)
    sc.set_paths(paths)


# save_path = "I:/4Multitask/exp/0ensamble\co_model11/ActiveMol_simlarity_output/"
test_path = "I:/4Multitask/exp/0ensamble\co_model11/test_output"


tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

test_listname = []
test_listname = sorted(listdir(test_path, test_listname))



label_list = []
y_Draw_list = []

for j in range(12):
    data = pd.read_csv(test_listname[j], usecols=['labels','T_score_mean'])

    scl = MinMaxScaler(feature_range=(-0.5,0.5))
    data['std_prob'] = scl.fit_transform(np.array(data['T_score_mean']).reshape(-1,1))
    d_class = pd.concat([abs(1 - data['std_prob']), abs(-1 - data['std_prob'])],axis=1)
    data['class_lag'] = d_class.min(axis=1)
    data['y_Draw'] = 1 - data['class_lag']
    data = shuffle(data)
    # data.to_csv('000.csv')
    label_list.append(data['labels'].to_list())
    y_Draw_list.append(data['y_Draw'].to_list())




scatter_size = 30
title_font = 20


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,13), dpi=150)
cm = plt.cm.get_cmap('RdYlBu')

sc1 = axes[0,0].scatter(range(len(y_Draw_list[0])), y_Draw_list[0],
                 c=y_Draw_list[0], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
# plt.colorbar(sc)
markset(sc1, label_list[0])
axes[0,0].set_title(tasks_name[0],fontsize=title_font)



sc1 = axes[0,1].scatter(range(len(y_Draw_list[1])), y_Draw_list[1],
                 c=y_Draw_list[1], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[1])
axes[0,1].set_title(tasks_name[1],fontsize=title_font)



sc1 = axes[0,2].scatter(range(len(y_Draw_list[2])), y_Draw_list[2],
                 c=y_Draw_list[2], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[2])
axes[0,2].set_title(tasks_name[2],fontsize=title_font)


sc1 = axes[1,0].scatter(range(len(y_Draw_list[3])), y_Draw_list[3],
                 c=y_Draw_list[3], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[3])
axes[1,0].set_title(tasks_name[3],fontsize=title_font)


sc1 = axes[1,1].scatter(range(len(y_Draw_list[4])), y_Draw_list[4],
                 c=y_Draw_list[4], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[4])
axes[1,1].set_title(tasks_name[4],fontsize=title_font)


sc1 = axes[1,2].scatter(range(len(y_Draw_list[5])), y_Draw_list[5],
                 c=y_Draw_list[5], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[5])
axes[1,2].set_title(tasks_name[5],fontsize=title_font)


sc1 = axes[2,0].scatter(range(len(y_Draw_list[6])), y_Draw_list[6],
                 c=y_Draw_list[6], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[6])
axes[2,0].set_title(tasks_name[6],fontsize=title_font)


sc1 = axes[2,1].scatter(range(len(y_Draw_list[7])), y_Draw_list[7],
                 c=y_Draw_list[7], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[7])
axes[2,1].set_title(tasks_name[7],fontsize=title_font)


sc1 = axes[2,2].scatter(range(len(y_Draw_list[8])), y_Draw_list[8],
                 c=y_Draw_list[8], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[8])
axes[2,2].set_title(tasks_name[8],fontsize=title_font)


sc1 = axes[3,0].scatter(range(len(y_Draw_list[9])), y_Draw_list[9],
                 c=y_Draw_list[9], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[9])
axes[3,0].set_title(tasks_name[9],fontsize=title_font)


sc1 = axes[3,1].scatter(range(len(y_Draw_list[10])), y_Draw_list[10],
                 c=y_Draw_list[10], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[10])
axes[3,1].set_title(tasks_name[10],fontsize=title_font)


sc1 = axes[3,2].scatter(range(len(y_Draw_list[11])), y_Draw_list[11],
                 c=y_Draw_list[11], vmin=0, vmax=0.5, s=scatter_size, cmap=cm)
markset(sc1, label_list[11])
axes[3,2].set_title(tasks_name[11],fontsize=title_font)

plt.tight_layout()
plt.savefig('Applicability Domain.png',dip=300)
plt.show()

