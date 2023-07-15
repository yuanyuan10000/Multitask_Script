import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as dist  # 导入scipy距离公式
from sklearn.utils import shuffle
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

save_path = "I:/4Multitask/exp/0ensamble\co_model11/ActiveMol_simlarity_output/"
test_path = "I:/4Multitask/exp/0ensamble\co_model11/test_output"


tasks_name = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

test_listname = []
test_listname = sorted(listdir(test_path, test_listname))


precision_list, recall_list, ap_list = [],[],[]
for j in range(12):

    data = pd.read_csv(test_listname[j])
    y_true = data['labels'].to_list()
    y_score = data['T_score_mean'].to_list()
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    precision_list.append(precision)
    recall_list.append(recall)
    ap_list.append(ap)





plt.figure(figsize=(10,8))
for i in range(7):
    plt.plot(recall_list[i],precision_list[i], linestyle="-",lw=2, label='{} (Area = {})'.format(tasks_name[i],round(ap_list[i],3)))
for i in range(7,12):
    plt.plot(recall_list[i],precision_list[i], linestyle="-.",lw=2, label='{} (Area = {})'.format(tasks_name[i],round(ap_list[i],3)))


plt.legend(fontsize=15)
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title('Precision-Recall curve',fontsize=20)
plt.xlabel('Recall',fontsize = 20)
plt.ylabel('Precision',fontsize = 20)
plt.tight_layout()
plt.savefig('PR_Curve.png',dip=300)
plt.show()

