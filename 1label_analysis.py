import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as dist  # 导入scipy距离公式
from sklearn.utils import shuffle
import pandas as pd
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

save_path = "I:/4Multitask/data/"
data_path = "I:/4Multitask/data/data"


tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

data_listname = []
data_listname = sorted(listdir(data_path, data_listname))

for i in range(12):
    locals()['df'+str(i)] = pd.read_csv(data_listname[i],index_col='smiles')

# mols = [Chem.MolFromSmiles(s) for s in df1['smiles'].to_list()]
# canonical_smi = [Chem.MolToSmiles(m) for m in mols]

data_1 = pd.concat([df0,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11],axis=1)
# data_1.reset_index(inplace=True)
# data_1.rename(columns={'index':'smiles'},inplace=True)
# df9.reset_index(inplace=True)
# df10.reset_index(inplace=True)
# df11.reset_index(inplace=True)
#
# data2= pd.merge(data_1,df9,how='outer',on='smiles')
# data3= pd.merge(data2,df10,how='outer',on='smiles')
# data4= pd.merge(data3,df11,how='outer',on='smiles')

data_1.to_csv('data_label_merge.csv')

#############################################################
# excel中计算标签数
import matplotlib.pyplot as plt
import brewer2mpl
import pandas as pd
from matplotlib import cm


data = pd.read_csv('data_label_merge.csv')

list = [(data['count']==1).sum(),(data['count']==2).sum(),(data['count']==3).sum(),
         (data['count']==4).sum(),(data['count']==5).sum(),(data['count']==6).sum(),
         (data['count']==7).sum(),(data['count']==8).sum(),(data['count']==9).sum(),
         (data['count']==10).sum(),(data['count']==11).sum(),(data['count']==12).sum()]


bmap = brewer2mpl.get_map('Set3', 'qualitative',12)
colors = bmap.mpl_colors

plt.figure(figsize=(10,8))
plt.bar(range(1,13),list,color=colors,edgecolor='#ABB6C8')
for x,y in zip(range(1,13),list):
    plt.text(x,y,y,ha='center',va='bottom',size=25,family="Times new roman") #文本注解 # 第一个参数是x轴坐标
plt.xticks(range(1,13),fontsize=25)
plt.yticks(fontsize=20)
plt.ylabel('Count',fontsize=25)
plt.title('Number of labels per molecule',fontsize=25)
plt.tight_layout()
plt.savefig('Number of labels per molecule.png',dip=300)
plt.show()




# left = pd.DataFrame({'A':['A1','A2','A3','A4'],'B':['B1','B2','B3','B4'],'key':['K1','K2','K3','K4']})
# right = pd.DataFrame({'C':['C1','C2','C3','C4'],'D':['D1','D2','D3','D4'],'key':['K1','K2','K5','K6']})
# a = pd.DataFrame({'C':['C5','C4','C3','C2'],'key':['K1','K2','K5','K6']})
