import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import urllib.request
from rdkit import Chem
from molvs import standardize_smiles
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools


df = pd.read_csv('I:\\2mol2vec\input_file\phd\pose_filter_lig_redup.csv')

df.columns = ['CID','smiles', 'label']


# 取出无效分子
none_list=[]
for i in range(df.shape[0]):
    if Chem.MolFromSmiles(df['smiles'][i]) is None:
        none_list.append(i)
df = df.drop(none_list,axis=0)
smiles = [standardize_smiles(i) for i in df['smiles'].values]
df['smiles'] = smiles
df.drop_duplicates(subset=['smiles','label'],keep='first',inplace=True)
df.drop_duplicates(subset=['smiles'],keep=False,inplace=True)

df.to_csv('C:/Users/201/Desktop/EGLN1_redup.csv', index=False)

smiles = []
for i in range(5155):
    print(i)
    a = standardize_smiles('O=C(c1cccc([N+](=O)[O-])c1)c1c([O-])[o+]c2ccccn12')
    smiles.append(a)