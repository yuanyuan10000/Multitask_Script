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
from ModelMixedNet import MixNet
from evaluate import model_evaluation, plot_confusion_matrix, plot_AUC


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


save_path = "I:/4Multitask/exp/result_mixnet/model13/"
train_path_des = 'I:/4Multitask/exp/D_desc/train'
test_path_des =  'I:/4Multitask/exp/D_desc/test'

train_path_fp = 'I:/4Multitask/exp/D_fp/train'
test_path_fp =  'I:/4Multitask/exp/D_fp/test'

epoch = 500
hidden_shared = 128
hidden = None
LR = 1e-3

# NOTE 每次跑程序一定要检查任务能不能对的上！！！
# # 任务文件名大写
# tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
#               'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# NOTE 每次跑程序一定要检查任务能不能对的上！！！
# 任务文件名小写
tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
tasks_num = len(tasks_name)

train_listname_des, test_listname_des = [],[]
sorted(listdir(train_path_des, train_listname_des))
sorted(listdir(test_path_des, test_listname_des))

train_listname_fp, test_listname_fp = [],[]
sorted(listdir(train_path_fp, train_listname_fp))
sorted(listdir(test_path_fp, test_listname_fp))

X_train_des, Y_train_des = [],[]
X_test_des, Y_test_des = [], []

X_train_fp, Y_train_fp = [],[]
X_test_fp, Y_test_fp = [], []

for i in range(tasks_num):
    x_train_des, y_train_des, x_test_des, y_test_des = get_train_test(train_listname_des[i], test_listname_des[i])
    X_train_des.append(torch.tensor(x_train_des).float())
    Y_train_des.append(torch.tensor(y_train_des).view(-1))
    X_test_des.append(torch.tensor(x_test_des).float())
    Y_test_des.append(torch.tensor(y_test_des).view(-1))

    x_train_fp, y_train_fp, x_test_fp, y_test_fp = get_train_test(train_listname_fp[i], test_listname_fp[i])
    X_train_fp.append(torch.tensor(x_train_fp).float())
    Y_train_fp.append(torch.tensor(y_train_fp).view(-1))
    X_test_fp.append(torch.tensor(x_test_fp).float())
    Y_test_fp.append(torch.tensor(y_test_fp).view(-1))

feature_size_des = X_train_des[0].shape[1] # 414
feature_size_fp = X_train_fp[0].shape[1] # 1191

statistics_index = copy.deepcopy(tasks_name)
statistics_index.append('Average')

loss_col = copy.deepcopy(tasks_name)
loss_col.append('AVERAGE')

average_loss = []
for i in range(tasks_num):
    locals()['loss' + str(i+1)] = []

torch.cuda.empty_cache()  # 释放GPU内存
torch.manual_seed(0)   # 设置随机种子


feature_size_des = X_train_des[0].shape[1]
feature_size_fp = X_train_fp[0].shape[1]


model= MixNet(feature_size_des, feature_size_fp,
              hidden_shared=hidden_shared, hidden=hidden, output=2, tasks_num=tasks_num)

optimizer = optim.Adam(model.parameters(), LR)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(1,epoch+1):
    train_outputs = model.mixnet_predict(X_train_des, X_train_fp, tasks_num)
    for i in range(tasks_num):
        locals()['l' + str(i + 1)] = criterion(train_outputs[i], Y_train_des[i])
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
test_outputs = model.mixnet_predict(X_test_des, X_test_fp, tasks_num)


train_preds_cls, train_labels = [], []
train_T_score, train_F_score = [], []
train_outputs = [F.softmax(output, dim=1) for output in train_outputs]
for i in range(tasks_num):
    train_preds_cls.append(train_outputs[i].argmax(-1).detach().numpy())
    train_labels.append(Y_train_des[i].numpy())
    pred_train = train_outputs[i].detach().numpy()
    train_T_score.append(pred_train[:, 1])
    train_F_score.append(pred_train[:, 0])


test_preds_cls, test_labels = [], []
test_T_score, test_F_score = [], []
test_outputs = [F.softmax(output, dim=1) for output in test_outputs]
for i in range(tasks_num):
    test_preds_cls.append(test_outputs[i].argmax(-1).detach().numpy())
    test_labels.append(Y_test_des[i].numpy())
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
loss_merge.to_csv(save_path+'Task Cost.csv')


# 保存模型
PATH = save_path+"MTDNN.pth"
torch.save(model.state_dict(), PATH)


# 训练集输出
for i in range(tasks_num):
    train_out=pd.DataFrame(np.hstack((train_labels[i].reshape(-1,1),train_preds_cls[i].reshape(-1,1),
                            train_T_score[i].reshape(-1,1),train_F_score[i].reshape(-1,1))))
    train_out.columns = ['train_labels','train_preds_cls','train_T_score','train_F_score']
    # train_out.to_excel(save_path+'train_{}_MTLnet_out.xlsx'.format(tasks_name[i]), header=True, index=False)
    train_out.to_csv(save_path+'train_{}_out.csv'.format(tasks_name[i]), header=True, index=False)

# 测试集输出
for i in range(tasks_num):
    test_out=pd.DataFrame(np.hstack((test_labels[i].reshape(-1,1),test_preds_cls[i].reshape(-1,1),
                            test_T_score[i].reshape(-1,1),test_F_score[i].reshape(-1,1))))
    test_out.columns = ['test_labels','test_preds_cls','test_T_score','test_F_score']
    # train_out.to_excel(save_path+'train_{}_MTLnet_out.xlsx'.format(tasks_name[i]), header=True, index=False)
    test_out.to_csv(save_path+'test_{}_out.csv'.format(tasks_name[i]), header=True, index=False)



# 训练集评估
for i in range(tasks_num):
    AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = \
        model_evaluation(train_labels[i], train_preds_cls[i], train_T_score[i])
    locals()["train_eval_" + str(i + 1)] = np.array([AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]).reshape(1,-1)
statistics_train = np.vstack([train_eval_1,train_eval_2,train_eval_3,train_eval_4,train_eval_5,train_eval_6,
                              train_eval_7,train_eval_8,train_eval_9,train_eval_10,train_eval_11,train_eval_12])
col_mean_train = statistics_train.mean(axis=0).reshape(1,-1)
statistics_train = np.vstack([statistics_train,col_mean_train])
statistics_train = pd.DataFrame(statistics_train)
statistics_train.columns = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
statistics_train.index = statistics_index
# statistics_train.to_csv(save_path+'MTLnet_train_Tox21_MACCS_statistics.csv',header=True,index=True)
statistics_train.to_excel(save_path+'MTDNN_train_Tox21_statistics.xlsx',header=True,index=True)

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
statistics_test.to_excel(save_path+'MTDNN_test_Tox21_statistics.xlsx',header=True,index=True)


# 训练集测试集平均值合并输出
col_mean = np.vstack([col_mean_train,col_mean_test])
col_Average = pd.DataFrame(col_mean,columns=["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"],
                                 index=['Train','Test'])
col_Average.to_excel(save_path+'MTDNN_col_Average_statistics.xlsx',header=True,index=True)

##########################################################################################################################
# 分别生成12个任务的混淆矩阵
for i in range(tasks_num):
    cnf_matrix = confusion_matrix(train_labels[i], train_preds_cls[i])
    class_names = np.array(['Negative', 'Positive'])
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm=cnf_matrix, classes=class_names,title='{} Confusion matrix'.format(tasks_name[i]))
    plt.savefig(save_path+"Train {} Confusion matrix".format(tasks_name[i]))
    # plt.show()


for i in range(tasks_num):
    cnf_matrix = confusion_matrix(test_labels[i], test_preds_cls[i])
    class_names = np.array(['Negative', 'Positive'])
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm=cnf_matrix, classes=class_names,title='{} Confusion matrix'.format(tasks_name[i]))
    plt.savefig(save_path+"Test {} test Confusion matrix".format(tasks_name[i]))
    # plt.show()

"""ROC曲线"""
plt.figure(figsize=(20,16))
plot_AUC(train_labels, train_T_score, tasks_name)
plt.title('ROC Curve', fontsize=30)
plt.savefig(save_path+"train_roc_curve.jpg")
plt.show()


plt.figure(figsize=(20,16))
plot_AUC(test_labels, test_T_score, tasks_name)
plt.title('ROC Curve', fontsize=30)
plt.savefig(save_path+"test_roc_curve.jpg")
plt.show()


