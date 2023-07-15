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
import torch.nn.functional as F
import copy
import os
from evaluate import model_evaluation, plot_confusion_matrix, plot_AUC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ModelMTGCN_test import MTEGCNPredictor,MTEGCNPredictor_2,MTGCNPredictor


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

save_path = "I:/4Multitask/exp/GCN_result/GCN/"
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

# # **********************************************************************************************************************
'训练模型'
torch.cuda.empty_cache()  # 释放GPU内存
torch.manual_seed(0)   # 设置随机种子


model = MTGCNPredictor(in_feats=n_feats, tasks_num=tasks_num)
# model = MTEGCNPredictor(in_feats=n_feats, dim_self_feat = dim_self_feat, tasks_num=tasks_num)
# model = MTEGCNPredictor_2(in_feats=n_feats, dim_self_feat = dim_self_feat, tasks_num=tasks_num)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

average_loss = []
for i in range(tasks_num):
    locals()['loss' + str(i+1)] = []

model.train()
for epoch in range(1,epoch+1):
    train_outputs = model(graph_train, atom_feats_train, tasks_num)
    # train_outputs = model(graph_train, atom_feats_train, self_feats_train, tasks_num)
    for i in range(tasks_num):
        locals()['l' + str(i + 1)] = criterion(train_outputs[i], Y_train[i])
        locals()["loss" + str(i+1)].append(locals()['l' + str(i + 1)].detach().item())
    Loss_ave = (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11 + l12) / 12
    average_loss.append(Loss_ave.detach().item())
    optimizer.zero_grad()
    Loss_ave.backward()
    optimizer.step()
    if epoch%10 == 0:
        print(f"epoch: {epoch}, AVERAGE_LOSS: {Loss_ave.detach().item():.3f}")

# 预测test的值
model.eval()
test_outputs = model(graph_test, atom_feats_test, tasks_num)
# test_outputs = model(graph_test, atom_feats_test, self_feats_test, tasks_num)

train_preds_cls, train_labels = [], []
train_T_score, train_F_score = [], []
train_outputs = [F.softmax(output, dim=1) for output in train_outputs]
for i in range(tasks_num):
    train_preds_cls.append(train_outputs[i].argmax(-1).detach().numpy())
    train_labels.append(Y_train[i].numpy())
    pred_train = train_outputs[i].detach().numpy()
    train_T_score.append(pred_train[:, 1])
    train_F_score.append(pred_train[:, 0])


test_preds_cls, test_labels = [], []
test_T_score, test_F_score = [], []
test_outputs = [F.softmax(output, dim=1) for output in test_outputs]
for i in range(tasks_num):
    test_preds_cls.append(test_outputs[i].argmax(-1).detach().numpy())
    test_labels.append(Y_test[i].numpy())
    pred_test = test_outputs[i].detach().numpy()
    test_T_score.append(pred_test[:, 1])
    test_F_score.append(pred_test[:, 0])


# 损失画图
plt.figure(figsize=(12, 8.5))
plt.plot(list(range(1,epoch+1)), average_loss[:epoch], label = 'Average Cost', linewidth=3, color = "darkred")
for i in range(7):
    plt.plot(list(range(1,epoch+1)), locals()["loss" + str(i+1)][:epoch], linestyle='--', label = tasks_name[i])
for i in range(7,12):
    plt.plot(list(range(1,epoch+1)), locals()["loss" + str(i+1)][:epoch], linestyle=':', label = tasks_name[i])
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylabel('Cost', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.xticks(list(range(0, epoch+2,50)), fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(save_path+"Task Cost.jpg")
plt.show()

### loss 结果输出
for i in range(tasks_num):
    locals()["loss" + str(i+1)] = np.array(locals()["loss" + str(i+1)]).reshape(-1,1)
average_loss = np.array(average_loss).reshape(-1,1)
loss_merge = np.hstack([loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10,loss11,loss12,average_loss])
loss_merge = pd.DataFrame(loss_merge)
loss_merge.columns = loss_col
loss_merge.to_csv(save_path+'Task Cost_{}.csv'.format(model_name))


# 保存模型
PATH = save_path+"{}.pth".format(model_name)
torch.save(model.state_dict(), PATH)

# 训练集输出
for i in range(tasks_num):
    train_out=pd.DataFrame(np.hstack((smiles_list_train[i].reshape(-1,1),train_labels[i].reshape(-1,1),train_preds_cls[i].reshape(-1,1),
                            train_T_score[i].reshape(-1,1),train_F_score[i].reshape(-1,1))))
    train_out.columns = ['smiles','train_labels','train_preds_cls','train_T_score','train_F_score']
    # train_out.to_excel(save_path+'train_{}_MTLnet_out.xlsx'.format(tasks_name[i]), header=True, index=False)
    train_out.to_csv(save_path+'train_{}_out.csv'.format(tasks_name[i]), header=True, index=False)

# 测试集输出
for i in range(tasks_num):
    test_out=pd.DataFrame(np.hstack((smiles_list_test[i].reshape(-1,1),test_labels[i].reshape(-1,1),test_preds_cls[i].reshape(-1,1),
                            test_T_score[i].reshape(-1,1),test_F_score[i].reshape(-1,1))))
    test_out.columns = ['smiles','test_labels','test_preds_cls','test_T_score','test_F_score']
    # train_out.to_excel(save_path+'train_{}_MTLnet_out.xlsx'.format(tasks_name[i]), header=True, index=False)
    test_out.to_csv(save_path + 'test_{}_out.csv'.format(tasks_name[i]), header=True, index=False)

# 训练集评估
for i in range(tasks_num):
    AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = \
        model_evaluation(train_labels[i], train_preds_cls[i], train_T_score[i])
    locals()["train_eval_" + str(i + 1)] = np.array([AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]).reshape(1,-1)
statistics_train = np.vstack([train_eval_1,train_eval_2,train_eval_3,train_eval_4,train_eval_5,train_eval_6,
                              train_eval_7,train_eval_8,train_eval_9,train_eval_10,train_eval_11,train_eval_12])
col_mean_train = statistics_train.mean(axis=0).reshape(1,-1)
statistics_train = np.vstack([statistics_train, col_mean_train])
statistics_train = pd.DataFrame(statistics_train)
statistics_train.columns = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
statistics_train.index = statistics_index
# statistics_train.to_csv(save_path+'MTLnet_train_Tox21_MACCS_statistics.csv',header=True,index=True)
statistics_train.to_excel(save_path+'MT{}_train_Tox21_statistics.xlsx'.format(model_name),header=True,index=True)

# 测试集评估
for i in range(tasks_num):
    AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = \
        model_evaluation(test_labels[i], test_preds_cls[i], test_T_score[i])
    locals()["test_eval_" + str(i + 1)] = np.array([AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]).reshape(1,-1)
statistics_test = np.vstack([test_eval_1,test_eval_2,test_eval_3,test_eval_4,test_eval_5,test_eval_6,
                              test_eval_7,test_eval_8,test_eval_9,test_eval_10,test_eval_11,test_eval_12])
col_mean_test = statistics_test.mean(axis=0).reshape(1,-1)
statistics_test = np.vstack([statistics_test,col_mean_test])
statistics_test = pd.DataFrame(statistics_test)
statistics_test.columns = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
statistics_test.index = statistics_index
# statistics_test.to_csv(save_path+'MTLnet_test_Tox21_MACCS_statistics.csv',header=True,index=True)
statistics_test.to_excel(save_path+'MT{}_test_Tox21_statistics.xlsx'.format(model_name),header=True,index=True)


# 训练集测试集平均值合并输出
col_mean = np.vstack([col_mean_train,col_mean_test])
col_Average = pd.DataFrame(col_mean,columns=["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"],
                                 index=['Train','Test'])
col_Average.to_excel(save_path+'MT{}_col_Average_statistics.xlsx'.format(model_name),header=True,index=True)

##########################################################################################################################
# 分别生成12个任务的混淆矩阵
for i in range(tasks_num):
    cnf_matrix = confusion_matrix(train_labels[i], train_preds_cls[i])
    class_names = np.array(['Negative', 'Positive'])
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm=cnf_matrix, classes=class_names,title='{} Confusion matrix'.format(tasks_name[i]))
    plt.savefig(save_path+"Train {} Confusion matrix".format(tasks_name[i]))
    plt.show()


for i in range(tasks_num):
    cnf_matrix = confusion_matrix(test_labels[i], test_preds_cls[i])
    class_names = np.array(['Negative', 'Positive'])
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm=cnf_matrix, classes=class_names,title='{} Confusion matrix'.format(tasks_name[i]))
    plt.savefig(save_path+"Test {} test Confusion matrix".format(tasks_name[i]))
    plt.show()

"""ROC曲线"""
plt.figure(figsize=(20,16))
plot_AUC(train_labels, train_T_score, tasks_name)
plt.title('ROC Curve', fontsize=30)
plt.savefig(save_path+"{}_train_roc_curve.jpg".format(model_name))
plt.show()


plt.figure(figsize=(20,16))
plot_AUC(test_labels, test_T_score, tasks_name)
plt.title('ROC Curve', fontsize=30)
plt.savefig(save_path+"{}_test_roc_curve.jpg".format(model_name))
plt.show()


