import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluate import model_evaluation, plot_confusion_matrix, plot_AUC
from sklearn.metrics import confusion_matrix

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


save_path = 'I:/4Multitask/exp/0ensamble/co_model11/'

desc_path = 'I:/4Multitask/exp/0ensamble/descriptions'
fp_path = 'I:/4Multitask/exp/0ensamble/fingerprints'
gcn_path = 'I:/4Multitask/exp/0ensamble/EGCN/model11'

# NOTE 每次跑程序一定要检查任务能不能对的上！！！
# 任务文件名小写
tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
statistics_index = copy.deepcopy(tasks_name)
statistics_index.append('Average')

desc_listname, fp_listname, gcn_listname = [],[],[]
desc_listname = sorted(listdir(desc_path, desc_listname))
fp_listname = sorted(listdir(fp_path, fp_listname))
gcn_listname = sorted(listdir(gcn_path, gcn_listname))

desc_list = [pd.read_csv(desc_listname[i]) for i in range(12)]
fp_list = [pd.read_csv(fp_listname[i]) for i in range(12)]
gcn_list = [pd.read_csv(gcn_listname[i]) for i in range(12)]


for i in range(12):
    print(i)
    assert (desc_list[i].shape[0]==fp_list[i].shape[0]==gcn_list[i].shape[0])

    result = pd.DataFrame(np.empty([desc_list[i].shape[0],8]))
    result.columns=['smiles','labels','labels_sum','preds_cls(>=1)','preds_cls(>=2)','preds_cls','T_score_mean','F_score_mean']
    result['smiles'] = gcn_list[i]['smiles']
    result['labels'] = desc_list[i]['test_labels']
    result['T_score_mean'] = (desc_list[i]['test_T_score']+fp_list[i]['test_T_score']+gcn_list[i]['test_T_score'])/3
    result['F_score_mean'] = (desc_list[i]['test_F_score']+fp_list[i]['test_F_score']+gcn_list[i]['test_F_score'])/3
    result['labels_sum'] = desc_list[i]['test_preds_cls']+fp_list[i]['test_preds_cls']+gcn_list[i]['test_preds_cls']


    a1 = result[result['labels_sum']>=2].reset_index()
    a1['preds_cls(>=2)'] = 1
    a2 = result[~(result['labels_sum']>=2)].reset_index()
    a2['preds_cls(>=2)'] = 0
    a = pd.concat([a1.drop(columns='index'), a2.drop(columns='index')])


    b1 = a[a['labels_sum']>=1].reset_index()
    b1['preds_cls(>=1)'] = 1
    b2 = a[~(a['labels_sum']>=1)].reset_index()
    b2['preds_cls(>=1)'] = 0
    b = pd.concat([b1.drop(columns='index'), b2.drop(columns='index')])


    c1 = b[b['T_score_mean']>=0.5].reset_index()
    c1['preds_cls'] = 1
    c2 = b[~(b['T_score_mean']>=0.5)].reset_index()
    c2['preds_cls'] = 0
    c = pd.concat([c1.drop(columns=['index']),c2.drop(columns=['index'])])


    locals()['co_result'+str(i)]  = c
    locals()['co_result'+str(i)].to_csv(save_path+tasks_name[i]+'.csv',index=False)

###################################################################################################


save_path23 = save_path+'vote23_'
save_path123 = save_path+'vote123_'
save_path_mean = save_path+'vote_mean_'
save_path_list = [save_path23, save_path123, save_path_mean]

labels, T_score_mean, preds_cls23, preds_cls123, preds_cls_mean = [],[],[],[],[]
for i in range(12):
    labels.append(np.array(locals()['co_result'+str(i)]['labels']))
    T_score_mean.append(np.array(locals()['co_result'+str(i)]['T_score_mean']))
    preds_cls_mean.append(np.array(locals()['co_result'+str(i)]['preds_cls']))
    preds_cls23.append(np.array(locals()['co_result'+str(i)]['preds_cls(>=2)']))
    preds_cls123.append(np.array(locals()['co_result'+str(i)]['preds_cls(>=1)']))
    assert (labels[i].shape == T_score_mean[i].shape == preds_cls23[i].shape == preds_cls123[i].shape == preds_cls_mean[i].shape)

preds_cls = [preds_cls23, preds_cls123, preds_cls_mean]

tasks_num = 12
# 训练集评估
for j in range(len(save_path_list)):
    for i in range(tasks_num):
        AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = model_evaluation(labels[i], preds_cls[j][i], T_score_mean[i])
        locals()["train_eval_" + str(i + 1)] = np.array([AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]).reshape(1,-1)
    statistics_train = np.vstack([train_eval_1,train_eval_2,train_eval_3,train_eval_4,train_eval_5,train_eval_6,
                                  train_eval_7,train_eval_8,train_eval_9,train_eval_10,train_eval_11,train_eval_12])
    col_mean_train = statistics_train.mean(axis=0).reshape(1,-1)
    statistics_train = np.vstack([statistics_train,col_mean_train])
    statistics_train = pd.DataFrame(statistics_train)
    statistics_train.columns = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
    statistics_train.index = statistics_index
    # statistics_train.to_csv(save_path+'MTLnet_train_Tox21_MACCS_statistics.csv',header=True,index=True)
    statistics_train.to_excel(save_path_list[j]+'MTDNN_train_Tox21_statistics.xlsx',header=True,index=True)



# # 分别生成12个任务的混淆矩阵
# for j in range(len(save_path_list)):
#     for i in range(tasks_num):
#         cnf_matrix = confusion_matrix(labels[i], preds_cls[j][i])
#         class_names = np.array(['Negative', 'Positive'])
#         np.set_printoptions(precision=2)
#
#         plt.figure(figsize=(10,8))
#         plot_confusion_matrix(cm=cnf_matrix, classes=class_names,title='{} Confusion matrix'.format(tasks_name[i]))
#         plt.savefig(save_path_list[j]+"Test {} test Confusion matrix".format(tasks_name[i]))
#         plt.show()


plt.figure(figsize=(20,16))
plot_AUC(labels, T_score_mean, tasks_name)
plt.title('ROC Curve', fontsize=30)
plt.savefig(save_path+"test_roc_curve.jpg")
plt.show()



# **********************************************************************************************************************


# #  仅有一个任务时的代码
#
#
# # NOTE 每次跑程序一定要检查任务能不能对的上！！！
# # 任务文件名小写
# tasks_name = ['NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
#               'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
# statistics_index = copy.deepcopy(tasks_name)
# statistics_index.append('Average')
#
# save_path = 'I:/4Multitask\exp/GCN_result/model8/'
# path = 'I:\\4Multitask\exp\GCN_result\\test'
# pathname=[]
# pathname = sorted(listdir(path, pathname))
# data_list = [pd.read_csv(pathname[i]) for i in range(12)]
# # data_list[0].head()
#
# labels, T_score, preds_cls = [],[],[]
# for i in range(12):
#     labels.append(np.array(data_list[i]['test_labels']))
#     T_score.append(np.array(data_list[i]['test_T_score']))
#     preds_cls.append(np.array(data_list[i]['test_preds_cls']))
#
#
# for i in range(tasks_num):
#     AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = \
#         model_evaluation(labels[i], preds_cls[i], T_score[i])
#     locals()["train_eval_" + str(i + 1)] = np.array([AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]).reshape(1,-1)
# statistics_train = np.vstack([train_eval_1,train_eval_2,train_eval_3,train_eval_4,train_eval_5,train_eval_6,
#                               train_eval_7,train_eval_8,train_eval_9,train_eval_10,train_eval_11,train_eval_12])
# col_mean_train = statistics_train.mean(axis=0).reshape(1,-1)
# statistics_train = np.vstack([statistics_train, col_mean_train])
# statistics_train = pd.DataFrame(statistics_train)
# statistics_train.columns = ["AUC", "ACC", "Recall(Sensitivity)", 'Specificity', 'BAC', "F1", "Kappa", "MCC", 'Precision', "BS"]
# statistics_train.index = statistics_index
# # statistics_train.to_csv(save_path+'MTLnet_train_Tox21_MACCS_statistics.csv',header=True,index=True)
# statistics_train.to_excel(save_path+'test_Tox21_statistics.xlsx',header=True,index=True)
#
#
# # # 分别生成12个任务的混淆矩阵
# # for i in range(tasks_num):
# #     cnf_matrix = confusion_matrix(labels[i], preds_cls[i])
# #     class_names = np.array(['Negative', 'Positive'])
# #     np.set_printoptions(precision=2)
# #
# #     plt.figure(figsize=(10,8))
# #     plot_confusion_matrix(cm=cnf_matrix, classes=class_names,title='{} Confusion matrix'.format(tasks_name[i]))
# #     plt.savefig(save_path+"Test {} Confusion matrix".format(tasks_name[i]))
# #     plt.show()
#
# """ROC曲线"""
# plt.figure(figsize=(20,16))
# plot_AUC(labels, T_score, tasks_name)
# plt.title('ROC Curve', fontsize=30)
# plt.savefig(save_path+"test_roc_curve.jpg")
# plt.show()