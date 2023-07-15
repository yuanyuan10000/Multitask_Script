import os
import pandas as pd
import numpy as np
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
import copy
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ModelMTDNN import MTLnet, MTLnet_2Layer, MTLnet_1Layer,MTLnet_2shared,MTLnet_3shared
from evaluate import model_evaluation, plot_confusion_matrix, plot_AUC


np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = True

class NET(nn.Module):
    def __init__(self, feature_size, shared_layer_size, tower_h1, tower_h2, output_size, dropout=0.5):
        super(NET, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=shared_layer_size),
            nn.Dropout(dropout),
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_h2, output_size)
        )

    def forward(self, x):
        out = self.sharedlayer(x)
        out = F.softmax(out,dim=1)
        return out


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

def get_data(filepath):
    # 取出smiles 计算指纹
    D = pd.read_csv(filepath)
    x_fp = D.drop(columns=['smiles',"Label"])
    x_fp = x_fp.values.astype(np.float64)
    x_des = copy.deepcopy(x_fp)
    y = pd.read_csv(filepath, usecols=['Label']).values
    return x_fp, x_des, y

def get_train_test(file_train, file_test):
    x_train_fp, x_train_des, y_train = get_data(file_train)
    x_test_fp, x_test_des, y_test = get_data(file_test)
    # 特征值标准化
    scl = StandardScaler()
    x_train_des = scl.fit_transform(x_train_des)
    x_test_des = scl.transform(x_test_des)
    # 将标准化前和标准化后的特征合并
    # x_train_des = np.hstack((x_train_fp, x_train_des))
    # x_test_des = np.hstack((x_test_fp, x_test_des))
    return x_train_des, y_train, x_test_des, y_test


save_path = "I:/4Multitask/exp/Fingerprints_result/DNN250epoch/"
train_path = 'I:/4Multitask/exp/Fingerprints/train'
test_path =  'I:/4Multitask/exp/Fingerprints/test'


epoch = 250
shared_layer_size = 512
tower_h1 = 256
tower_h2 = 128
output_size = 2
LR = 1e-3
# NOTE 每次跑程序一定要检查任务能不能对的上！！！
# # 任务文件名大写
# tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
#               'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# NOTE 每次跑程序一定要检查任务能不能对的上！！！
# 任务文件名小写
tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
statistics_index = copy.deepcopy(tasks_name)
statistics_index.append('Average')
tasks_num = len(tasks_name)

train_listname, test_listname = [],[]
train_listname = sorted(listdir(train_path, train_listname))
test_listname = sorted(listdir(test_path, test_listname))

loss_col = copy.deepcopy(tasks_name)
loss_col.append('AVERAGE')
X_train, Y_train = [], []
X_test, Y_test = [], []


for i in range(tasks_num):
    x_train_des, y_train, x_test_des, y_test = get_train_test(train_listname[i], test_listname[i])

    X_train.append(torch.tensor(x_train_des).float())
    Y_train.append(torch.tensor(y_train).view(-1))
    X_test.append(torch.tensor(x_test_des).float())
    Y_test.append(torch.tensor(y_test).view(-1))


statistics_test = []

for j in range(tasks_num):

    print('正在处理任务{}'.format(tasks_name[j]))

    torch.cuda.empty_cache()  # 释放GPU内存
    torch.manual_seed(0)   # 设置随机种子

    feature_size = X_train[0].shape[1]
    model = NET(feature_size, shared_layer_size, tower_h1, tower_h2, output_size)

    optimizer = optim.Adam(model.parameters(), LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for i in range(epoch):
        train_output = model(X_train[j])
        loss = criterion(train_output,Y_train[j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print('epoch:{}\tloss:{}'.format(i,loss))


    model.eval()
    test_output = model(X_test[j])

    # test_output = F.softmax(test_output,dim=1)
    test_pred_cls = test_output.detach().argmax(-1).numpy()
    pred_test = test_output.detach().numpy()
    test_label = Y_test[j].numpy()
    test_T_score = pred_test[:,1]
    tets_F_score = pred_test[:,0]

    path = save_path + 'DNN_{}.pth'.format(tasks_name[j])
    torch.save(model.state_dict(),path)

    AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = \
            model_evaluation(test_label, test_pred_cls, test_T_score)
    test_eval = [AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]
    statistics_test.append(test_eval)

col = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
statistics_test = np.array(statistics_test)
statistics_mean = statistics_test.mean(axis=0)
statistics = pd.DataFrame(np.vstack([statistics_test,statistics_mean]))
statistics.columns = col
statistics.index = statistics_index
statistics.to_csv(save_path+'test_statistic.csv')
