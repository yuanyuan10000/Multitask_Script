from dgllife.utils import *
import dgl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import *
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import numpy as np
import pandas as pd
import urllib.request
from smiles2graph import complete_graph, bigraph, nearest_neighbor_graph
from ModelGCN import GCNPredictor
import torch.nn.functional as F
import copy
import os
from evaluate import model_evaluation, plot_confusion_matrix, plot_AUC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = True


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


def collate_emodel(samples, dim_self_feat):
    self_feats = np.empty((len(samples), dim_self_feat),dtype=np.float32)   # (5311, 3)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings

        # self_feats[i, 8] = mol_graph.max_abs_charge
        # self_feats[i, 9] = mol_graph.min_abs_charge
        # self_feats[i, 10] = mol_graph.num_rad_elc
        # self_feats[i, 11] = mol_graph.num_val_elc
        #
        self_feats[i, 3] = mol_graph.mol_logP
        self_feats[i, 4] = mol_graph.NHA
        self_feats[i, 5] = mol_graph.NHD
        self_feats[i, 6] = mol_graph.num_rotatable_bonds
        self_feats[i, 7] = mol_graph.qed_score

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(self_feats), torch.tensor(labels).view(-1)


def data_load(tasks_num, listname, dim_self_feat):
    X, Y, self_feats,smiles_list = [],[],[],[]
    for i in range(tasks_num):
        print('正在处理文件{}'.format(os.path.basename(listname[i])))
        dataset, n_feats, smiles = bigraph(listname[i])
        graph, self_feat, labels = collate_emodel(dataset, dim_self_feat)
        X.append(graph)
        Y.append(labels)
        smiles_list.append(smiles)
        self_feats.append(self_feat)
    return X, self_feats, Y, n_feats, smiles_list

# # **********************************************************************************************************************
"处理数据"

save_path = "I:/4Multitask/exp/GCN_result/GCN_out/"
train_path = 'I:/4Multitask/exp/Fingerprints/train'
test_path =  'I:/4Multitask/exp/Fingerprints/test'
model_name = 'GCN'

# self_feat：Graph Features(m, w)
dim_self_feat = 8

epoch = 500

# NOTE 每次跑程序一定要检查任务能不能对的上！！！
# 任务文件名小写
tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

train_listname, test_listname = [],[]
train_listname = sorted(listdir(train_path, train_listname))
test_listname = sorted(listdir(test_path, test_listname))

statistics_index = copy.deepcopy(tasks_name)
statistics_index.append('Average')
tasks_num = len(tasks_name)

loss_col = copy.deepcopy(tasks_name)
loss_col.append('AVERAGE')


graph_train, self_feats_train, Y_train, n_feats_train,smiles_list_train = data_load(tasks_num, train_listname, dim_self_feat)
graph_test, self_feats_test, Y_test, n_feats_test, smiles_list_test = data_load(tasks_num, test_listname, dim_self_feat)


# n_feats: 节点特征size
lengths = [n_feats_train, n_feats_test]
assert len(set(lengths)) == 1
n_feats = n_feats_train

# atom_feats: 节点特征
atom_feats_train = [graph.ndata.pop('feat') for graph in graph_train]
atom_feats_test = [graph.ndata.pop('feat') for graph in graph_test]


#####################################################################

statistics_test = []

for j in range(tasks_num):
    print('正在处理任务{}'.format(tasks_name[j]))

    torch.cuda.empty_cache()
    torch.manual_seed(0)

    model = GCNPredictor(in_feats=n_feats)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    creterion = CrossEntropyLoss()

    model.train()
    for i in range(epoch):
        train_output = model(graph_train[j],atom_feats_train[j])
        loss = creterion(train_output,Y_train[j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print('epoch:{}\tloss:{}'.format(i,loss))


    model.eval()
    test_output = model(graph_test[j],atom_feats_test[j])

    test_output = F.softmax(test_output,dim=1)
    test_pred_cls = test_output.detach().argmax(-1).numpy()
    pred_test = test_output.detach().numpy()
    test_label = Y_test[j].numpy()
    test_T_score = pred_test[:, 1]
    tets_F_score = pred_test[:, 0]

    path = save_path + 'DNN_{}.pth'.format(tasks_name[j])
    torch.save(model.state_dict(), path)

    AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = \
        model_evaluation(test_label, test_pred_cls, test_T_score)
    test_eval = [AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]
    statistics_test.append(test_eval)

col = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
statistics_test = np.array(statistics_test)
statistics_mean = statistics_test.mean(axis=0)
statistics = pd.DataFrame(np.vstack([statistics_test, statistics_mean]))
statistics.columns = col
statistics.index = statistics_index
statistics.to_csv(save_path + 'test_statistic.csv')