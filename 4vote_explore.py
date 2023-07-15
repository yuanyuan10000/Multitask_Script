import numpy as np
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
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


def try_prob(prob, desc_list, fp_list, gcn_list):
    labels, preds_cls, T_score = [], [], []
    for i in range(12):
        result = pd.DataFrame(np.empty([desc_list[i].shape[0], 3]))
        result.columns = ['labels', 'preds_cls', 'T_score']
        result['labels'] = desc_list[i]['test_labels']
        result['T_score'] = (desc_list[i]['test_T_score'] + fp_list[i]['test_T_score'] + gcn_list[i]['test_T_score']) / 3

        c1 = result[result['T_score'] >= prob].reset_index()
        c1['preds_cls'] = 1
        c2 = result[~(result['T_score'] >= prob)].reset_index()
        c2['preds_cls'] = 0
        c = pd.concat([c1.drop(columns=['index']), c2.drop(columns=['index'])])

        labels.append(np.array(np.array(c['labels'])))
        preds_cls.append(np.array(c['preds_cls']))
        T_score.append(np.array(c['T_score']))

    return labels, preds_cls, T_score


save_path = 'I:/4Multitask/exp/0ensamble/co_model1/'

desc_path = 'I:/4Multitask/exp/0ensamble/descriptions'
fp_path = 'I:/4Multitask/exp/0ensamble/fingerprints'
gcn_path = 'I:/4Multitask/exp/0ensamble/EGCN'

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

prob_list = (np.array(range(10,51,2))/100).tolist()

for i in range(12):
    assert (desc_list[i].shape[0] == fp_list[i].shape[0] == gcn_list[i].shape[0])


acc_list, sensitivity_list, specificity_list, BAC_list, f1_list, kappa_list, mcc_list, precision_list = [],[],[],[],[],[],[],[]
for p in prob_list:
    labels, preds_cls,T_score = try_prob(p, desc_list, fp_list, gcn_list)

    for i in range(12):
        AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS = model_evaluation(labels[i], preds_cls[i], T_score[i])
        locals()["train_eval_" + str(i + 1)] = np.array([AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]).reshape(1, -1)
    statistics = np.vstack([train_eval_1, train_eval_2, train_eval_3, train_eval_4, train_eval_5, train_eval_6,
                                  train_eval_7, train_eval_8, train_eval_9, train_eval_10, train_eval_11,train_eval_12])
    col_mean = statistics.mean(axis=0).tolist()

    acc_list.append(col_mean[1])
    sensitivity_list.append(col_mean[2])
    specificity_list.append(col_mean[3])
    BAC_list.append(col_mean[4])
    f1_list.append(col_mean[5])
    kappa_list.append(col_mean[6])
    mcc_list.append(col_mean[7])
    precision_list.append(col_mean[8])



plt.figure(figsize=(10,8))
plt.plot(prob_list,acc_list,label='ACC',linewidth=5)
plt.plot(prob_list,sensitivity_list,label='sensitivity',linewidth=5)
plt.plot(prob_list,specificity_list,label='specificity',linewidth=5)
plt.plot(prob_list,BAC_list,label='BAC',linewidth=5)
plt.plot(prob_list,f1_list,label='F1',linewidth=5)
plt.plot(prob_list,kappa_list,label='kappa',linewidth=5)
plt.plot(prob_list,mcc_list,label='MCC',linewidth=5)
plt.plot(prob_list,precision_list,label='precision',linewidth=5)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()