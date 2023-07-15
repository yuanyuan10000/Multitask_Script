import os
import pandas as pd
import numpy as np


# 取出smiles和Label合并
file_list=os.listdir('./exp/desc/train')
file_path = ["./exp/desc/train/"+str(i) for i in file_list]

for i in range(12):
    df = pd.read_csv(file_path[i],usecols=['smiles','Label'])
    df.to_csv('./exp/D_test/'+file_list[i][:-22]+'.csv', index=None)





# 取出指定的描述符或指纹，与smiles和label进行合并
file_list = os.listdir('./exp/fp/train')
file_path = ["./exp/fp/train/"+str(i) for i in file_list]

for i in range(12):
    df = pd.read_csv(file_path[i])

    smiles = df.iloc[:,0]
    label = df.iloc[:,-1]
    subset1 = df.iloc[:,1:1025]
    subset2 = df.iloc[:,1025:-1]

    data1 = pd.concat([smiles,subset1,label],axis=1)
    data2 = pd.concat([smiles,subset2,label],axis=1)

    data1.to_csv('exp/ECFP6/train/'+file_list[i][:-16]+'_ECFP6.csv',index=None)
    data2.to_csv('exp/MACCS/train/'+file_list[i][:-16]+'_MACCS.csv', index=None)


