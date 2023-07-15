import os, sys
import re
import hashlib
import shutil
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import six as _six

################################ 合并x和label #####################################
# # os.mkdir("MACCS1")
#
# x_file_list=os.listdir('MACCS')
# x_file_root = ["MACCS/"+str(i) for i in x_file_list]
# y_file_list=os.listdir('data_redup')
# y_file_root = ["0data_redup/"+str(i) for i in y_file_list]
#
# file_root = list(map(lambda x,y: [x,y], x_file_root, y_file_root))
# print(file_root)
#
# for x_file, y_file in file_root:
#     print(x_file)
#     print(y_file)
#     x = pd.read_csv(x_file)
#     y = pd.read_csv(y_file)
#     y = y.iloc[:,-1]
#     data = pd.concat([x, y], axis=1)
#     data.to_csv("MACCS1/" + x_file[6:],index=False)


############################ 将12个数据集的3个特征合并到一起 ####################################

file_list_1 = os.listdir('./data/MACCS')
file_root_1 = ['./data/MACCS/'+str(i) for i in file_list_1]

file_list_2 = os.listdir('./data/ECFP6')
file_root_2 = ["./data/ECFP6/"+str(i) for i in file_list_2]

file_list_3=os.listdir('./data/RDKit2D')
file_root_3 = ["./data/RDKit2D/"+str(i) for i in file_list_3]

file_list_4=os.listdir('./data/mordred3D')
file_root_4 = ["./data/mordred3D/"+str(i) for i in file_list_4]


# 指纹
for i in range(len(file_list_1)):
    assert file_list_1[i][:-10]==file_list_2[i][:-10]
    print("正在处理第{}个文件".format(i+1))
    data1 = pd.read_csv(file_root_1[i],index_col='smiles').iloc[:,:-1]
    data2 = pd.read_csv(file_root_2[i],index_col='smiles')
    fp = pd.concat([data1,data2],axis=1)
    chem_num1 = fp.shape[0]
    assert data1.shape[0] == data2.shape[0] == fp.shape[0]
    # 缺失值处理
    fp = fp[~fp.isin([np.nan, np.inf, -np.inf]).any(1)]
    chem_num2 = fp.shape[0]
    print("已删除{}文件中含有空值或无穷大值的化合物{}个".format(file_root_1[i][9:-4], (chem_num1 - chem_num2)))
    fp.to_csv('./data/fingerprints/'+ file_list_1[i][:-4] + '_ECFP6.csv')
    print("----------------- end -----------------")

# 描述符
for i in range(len(file_list_3)):
    assert file_list_3[i][:-13]==file_list_4[i][:-14]
    print("正在处理第{}个文件".format(i+1))
    data3 = pd.read_csv(file_root_3[i],index_col='smiles').iloc[:,:-1]
    data4 = pd.read_csv(file_root_4[i],index_col='smiles')
    desc = pd.concat([data3,data4],axis=1)
    chem_num1 = desc.shape[0]
    assert data3.shape[0]==data4.shape[0]==desc.shape[0]
    # 缺失值处理
    desc = desc[~desc.isin([np.nan, np.inf, -np.inf]).any(1)]
    chem_num2 = desc.shape[0]
    print("已删除{}文件中含有空值或无穷大值的化合物{}个".format(file_root_3[i][15:-4], (chem_num1 - chem_num2)))
    desc.to_csv('./data/descriptions/'+ file_list_3[i][:-4] + '_mordred3D.csv')
    print("----------------- end -----------------")







