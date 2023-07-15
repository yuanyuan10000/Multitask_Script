import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as dist  # 导入scipy距离公式
from sklearn.utils import shuffle
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


save_path = "I:/4Multitask/exp/Fingerprints_result\MMT-DNN\ActiveMol_simlarity_output/"
test_path = "I:\\4Multitask\exp\\0ensamble\\fingerprints"
train_path = "I:/4Multitask/exp/0ensamble/train_smi"

tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

train_listname,test_listname = [],[]
train_listname = sorted(listdir(train_path, train_listname))
test_listname = sorted(listdir(test_path, test_listname))



for j in range(12):
    print('正在处理任务: {}'.format(tasks_name[j]))
    df_train = pd.read_csv(train_listname[j])
    train_smi = np.ravel(df_train[df_train['Label']==1]['smiles'])
    # train_smi = np.ravel(pd.read_csv(train_listname[j],usecols=['smiles']))
    train_mols = [Chem.MolFromSmiles(m) for m in train_smi]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in train_mols]

    df_test = pd.read_csv(test_listname[j])
    df_test = df_test[df_test['test_labels']==1].reset_index().drop(columns=['index'])
    test_smi = np.ravel(df_test['smiles'])
    test_mols = [Chem.MolFromSmiles(m) for m in test_smi]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2,1024) for mol in test_mols]

    dist = []
    for i in range(len(test_mols)):
        simlarity = np.array(DataStructs.BulkTanimotoSimilarity(test_fps[i], train_fps)).max()
        dist.append(simlarity)

    df_test['smilarity'] = dist
    df_test.to_csv(save_path+tasks_name[j]+'.csv',index=False)


##############################################################################################################################333

tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']


save_path = "I:\\4Multitask\exp\\0ensamble\co_model11/"
data_path = "I:\\4Multitask\exp\\0ensamble\co_model11\ActiveMol_simlarity_output"

data_listname = []
data_listname = sorted(listdir(data_path, data_listname))
#
# p_list = []
#
# for j in range(12):
#
#     data = pd.read_csv(data_listname[j])
#     data['result'] = (data['test_labels']==data['test_preds_cls']).to_list()
#
#     try:
#         p1 = (data[data['smilarity']>=0.8]['result'].sum())/(data[data['smilarity']>=0.8].shape[0])
#     except ZeroDivisionError:
#         p1 = 'ZeroDivision'
#
#     try:
#         p2 = (data[((data['smilarity']<0.8)*(data['smilarity']>=0.6))]['result'].sum())/(data[((data['smilarity']<0.8)*(data['smilarity']>=0.6))].shape[0])
#     except ZeroDivisionError:
#         p2 = 'ZeroDivision'
#
#     try:
#         p3 = (data[((data['smilarity']<0.6)*(data['smilarity']>=0.4))]['result'].sum())/(data[((data['smilarity']<0.6)*(data['smilarity']>=0.4))].shape[0])
#     except ZeroDivisionError:
#         p3 = 'ZeroDivision'
#
#
#     try:
#         p4 = (data[data['smilarity']<0.4]['result'].sum())/(data[data['smilarity']<0.4].shape[0])
#     except ZeroDivisionError:
#         p4 = 'ZeroDivision'
#
#     p = [p1,p2,p3,p4]
#     p_list.append(p)
#
# df_prob = pd.DataFrame(p_list)
# df_prob.index = tasks_name
# df_prob.columns = ['>=0.8','<0.8','<0.6','<0.4']
# df_prob.to_excel(save_path+'不同相似性阈值下正确预测活性分子百分比.xlsx')


p_list = []

for j in range(12):

    data = pd.read_csv(data_listname[j])
    data['result'] = (data['labels']==data['preds_cls']).to_list()

    try:
        p1 = (data[data['smilarity']>=0.8]['result'].sum())/(data[data['smilarity']>=0.8].shape[0])
    except ZeroDivisionError:
        p1 = 'ZeroDivision'

    try:
        p2 = (data[((data['smilarity']<0.8)*(data['smilarity']>=0.6))]['result'].sum())/(data[((data['smilarity']<0.8)*(data['smilarity']>=0.6))].shape[0])
    except ZeroDivisionError:
        p2 = 'ZeroDivision'

    try:
        p3 = (data[((data['smilarity']<0.6)*(data['smilarity']>=0.4))]['result'].sum())/(data[((data['smilarity']<0.6)*(data['smilarity']>=0.4))].shape[0])
    except ZeroDivisionError:
        p3 = 'ZeroDivision'


    try:
        p4 = (data[data['smilarity']<0.4]['result'].sum())/(data[data['smilarity']<0.4].shape[0])
    except ZeroDivisionError:
        p4 = 'ZeroDivision'

    p = [(data[data['smilarity']>=0.8].shape[0]), (data[data['smilarity']>=0.8]['result'].sum()), p1,
         (data[((data['smilarity']<0.8)*(data['smilarity']>=0.6))].shape[0]), (data[((data['smilarity']<0.8)*(data['smilarity']>=0.6))]['result'].sum()), p2,
         (data[((data['smilarity']<0.6)*(data['smilarity']>=0.4))].shape[0]), (data[((data['smilarity']<0.6)*(data['smilarity']>=0.4))]['result'].sum()), p3,
         (data[data['smilarity']<0.4].shape[0]), (data[data['smilarity']<0.4]['result'].sum()), p4]
    p_list.append(p)

df_prob = pd.DataFrame(p_list)
df_prob.index = tasks_name
# df_prob.columns = ['>=0.8','<0.8','<0.6','<0.4']
df_prob.to_excel(save_path+'不同相似性阈值下正确预测活性分子百分比.xlsx')