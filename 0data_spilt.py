import os, sys
import re
import hashlib
import shutil
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import six as _six
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

################################### 划分数据集 ####################################
# 按照4:1的比例划分训练集和测试集
def SpiltD(data,train_idx):
    train_data = data[data.index.isin(train_idx)]
    test_data = data[~data.index.isin(train_idx)]
    return train_data,test_data


filepath_fp = []
filepath_desc = []
filepath_fp = sorted(listdir('I:/4Multitask/data/fingerprints', filepath_fp))
filepath_desc = sorted(listdir('I:/4Multitask/data/descriptions', filepath_desc))

task_num = len(filepath_fp)
list = []
index = []
for i in range(task_num):
    print('正在处理文件{}'.format(os.path.basename(filepath_fp[i])[:-16]))
    data_fp = shuffle(pd.read_csv(filepath_fp[i]),random_state=23)
    data_desc = shuffle(pd.read_csv(filepath_desc[i]),random_state=23)

    neg_fp = data_fp.loc[data_fp['Label'] == 0]
    pos_fp = data_fp.loc[data_fp['Label'] == 1]
    neg_desc = data_desc.loc[data_desc['Label'] == 0]
    pos_desc = data_desc.loc[data_desc['Label'] == 1]

    train_idx = data_fp.sample(frac=0.8, random_state=11).index

    NegTrain_fp, NegTest_fp = SpiltD(neg_fp, train_idx)
    PosTrain_fp, PosTest_fp = SpiltD(pos_fp, train_idx)
    NegTrain_desc, NegTest_desc = SpiltD(neg_desc, train_idx)
    PosTrain_desc, PosTest_desc = SpiltD(pos_desc, train_idx)

    train_fp = shuffle(pd.concat([NegTrain_fp, PosTrain_fp]),random_state=23)
    test_fp = shuffle(pd.concat([NegTest_fp, PosTest_fp]),random_state=23)
    train_desc = shuffle(pd.concat([NegTrain_desc, PosTrain_desc]),random_state=23)
    test_desc = shuffle(pd.concat([NegTest_desc, PosTest_desc]),random_state=23)

    train_fp.to_csv("fp/train/train_" + os.path.basename(filepath_fp[i]), index=False)
    test_fp.to_csv("fp/test/test_" + os.path.basename(filepath_fp[i]), index=False)
    train_desc.to_csv("desc/train/train_" + os.path.basename(filepath_desc[i]), index=False)
    test_desc.to_csv("desc/test/test_" + os.path.basename(filepath_desc[i]), index=False)

    print("共有分子{}个,其中正样本个数为{}个，负样本个数为{}个".format(
        data_fp.shape[0],pos_fp.shape[0],neg_fp.shape[0]))
    print("训练集分子为{},其中正样本个数为{}个，负样本个数为{}个".format(
        train_fp.shape[0],PosTrain_fp.shape[0], NegTrain_fp.shape[0]))
    print("测试集分子为{},其中正样本个数为{}个，负样本个数为{}个".format(
        test_fp.shape[0], PosTest_fp.shape[0], NegTest_fp.shape[0]))
    print('文件{}处理成功'.format(os.path.basename(filepath_fp[0])[:-16]))
    print("*"*100)

    list.append([data_fp.shape[0], pos_fp.shape[0], neg_fp.shape[0],
                 train_fp.shape[0], PosTrain_fp.shape[0],  NegTrain_fp.shape[0],
                 test_fp.shape[0], PosTest_fp.shape[0], NegTest_fp.shape[0]])
    index.append(os.path.basename(filepath_fp[i])[:-21])


list_D = pd.DataFrame(list,columns=['Total','Positive','Negative',
                                    'Train','Train Pos','Train Neg',
                                    'Test','Test Pos','Test Neg'],
                      index=index)
list_D.to_csv("data statistics.csv")



"""# 按照active和inactive的样本量分层采样
def SpiltD(data):
    train_idx = data.sample(frac=0.8,random_state=11).index
    train_data = data[data.index.isin(train_idx)]
    test_data = data[~data.index.isin(train_idx)]
    return train_data,test_data

filepath = []
filepath = sorted(listdir('I:/4Multitask/data/fingerprint', filepath))
# print(filepath)
task_num = len(filepath)

# 按照4:1的比例划分训练集和测试集

for file in filepath:
    data =  shuffle(pd.read_csv(file),random_state=23)
    neg_D = data.loc[data['Label'] == 0]
    pos_D = data.loc[data['Label'] == 1]

    NegTrain_D, NegTest_D = SpiltD(neg_D)
    PosTrain_D, PosTest_D = SpiltD(pos_D)

    train = pd.concat([NegTrain_D,PosTrain_D])
    train = shuffle(train)
    test = pd.concat([NegTest_D,PosTest_D])
    test = shuffle(test)

    train.to_csv("train/noSampling/train_" + os.path.basename(file), index=False)
    test.to_csv("test/noSampling/test_" + os.path.basename(file), index=False)
    print('文件{}处理成功'.format(os.path.basename(file)))
"""
