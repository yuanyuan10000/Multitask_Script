import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline


def smooth_xy(lx, ly):
    """
    数据平滑处理
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(),100)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]



eva_EGCN = pd.read_excel("I:/4Multitask/exp/GCN_result\MMT-EGCN\model11\MTGCN_test_Tox21_statistics.xls").iloc[:-1,:]
eva_desc = pd.read_excel("I:/4Multitask/exp/Descriptors_result\MMT-DNN\MTDNN_test_Tox21_statistics.xls").iloc[:-1,:]
eva_fp = pd.read_excel("I:/4Multitask/exp/Fingerprints_result\MMT-DNN\MTDNN_test_Tox21_statistics.xls").iloc[:-1,:]
co_model = pd.read_excel("I:/4Multitask/exp/0ensamble\co_model11/vote_mean_test_statistics.xls").iloc[:-1,:]
eva_GCN = pd.read_excel("I:/4Multitask\exp\GCN_result\MMT-GCN\MTGCN_test_Tox21_statistics.xls").iloc[:-1,:]




# col = ['Co-Model','Des MMT-DNN','FP MMT-DNN','MMT-EGCN']
col = ['Co-Model','Des MMT-DNN','FP MMT-DNN','MMT-GCN','MMT-EGCN']
df_auc = pd.concat([co_model['AUC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_desc['AUC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_fp['AUC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_GCN['AUC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_EGCN['AUC'].sort_values(ascending=False).reset_index().drop(columns=['index'])],axis=1)
df_auc.columns=col

#---------- 总体预测性能比较：小提琴图 ---------
# sns.set(style="whitegrid")
# # 以下两句防止中文显示为窗格plt.rcParams["font.sans-serif"]=["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# 设置窗口的大小
f, ax = plt.subplots(figsize=(11.5, 6))
# 绘制小提琴图
g = sns.violinplot(data=df_auc, palette="Set2", bw=.35, cut=2, linewidth=2)
# 设置轴显示的范围
#ax.set(ylim=(-.7, 1.05))
# 去除上下左右的边框（默认该函数会取出右上的边框）
sns.despine(left=True, bottom=True)
# ax.set_title('AUC',fontsize=25,fontweight='bold')
ax.set_ylabel('AUC',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('./exp/AUC violinplot.png',dpi=300)
plt.show()

#---------- 总体预测性能比较：折线图 ---------
lw = 3
plt.figure(figsize=(10,8))

# x = range(0,12)
# x_s,y_s0 = smooth_xy(x,df_auc[col[0]])
# x_s,y_s1 = smooth_xy(x,df_auc[col[1]])
# x_s,y_s2 = smooth_xy(x,df_auc[col[2]])
# x_s,y_s3 = smooth_xy(x,df_auc[col[3]])
# x_s,y_s4 = smooth_xy(x,df_auc[col[4]])
#
# plt.plot(y_s0,lw=lw,label=col[0])
# plt.plot(y_s1,lw=lw,label=col[1])
# plt.plot(y_s2,lw=lw,label=col[2])
# plt.plot(y_s3,lw=lw,label=col[3])
# plt.plot(y_s4,lw=lw,label=col[4])

plt.plot(df_auc['Co-Model'],lw=lw,label='Co-Model')
plt.plot(df_auc['Des MMT-DNN'],lw=lw,label='Des MMT-DNN')
plt.plot(df_auc['FP MMT-DNN'],lw=lw,label='FP MMT-DNN')
plt.plot(df_auc['MMT-EGCN'],lw=lw,label='MMT-EGCN')
plt.plot(df_auc['MMT-GCN'],lw=lw,label='MMT-GCN')
plt.grid(False)
plt.xticks([])
plt.xlim(0,11)
plt.yticks(fontsize=20)
plt.ylabel('AUC',fontsize=25)
plt.legend(fontsize=20,loc= 'lower left')
plt.savefig('./exp/AUC lineplot.png',dpi=300)
plt.show()


#######################################################################################################

df_acc = pd.concat([co_model['ACC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_desc['ACC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_fp['ACC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_GCN['ACC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_EGCN['ACC'].sort_values(ascending=False).reset_index().drop(columns=['index'])],axis=1)
df_acc.columns=col

#---------- 总体预测性能比较：小提琴图 ---------
f, ax = plt.subplots(figsize=(11.5, 6))
g = sns.violinplot(data=df_acc, palette="Set2", bw=.35, cut=2, linewidth=2)
sns.despine(left=True, bottom=True)
ax.set_ylabel('ACC',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('./exp/ACC violinplot.png',dpi=300)
plt.show()

#---------- 总体预测性能比较：折线图 ---------
lw = 3
plt.figure(figsize=(10,8))
plt.plot(df_acc['Co-Model'],lw=lw,label='Co-Model')
plt.plot(df_acc['Des MMT-DNN'],lw=lw,label='Des MMT-DNN')
plt.plot(df_acc['FP MMT-DNN'],lw=lw,label='FP MMT-DNN')
plt.plot(df_acc['MMT-EGCN'],lw=lw,label='MMT-EGCN')
plt.plot(df_acc['MMT-GCN'],lw=lw,label='MMT-GCN')
plt.grid(False)
plt.xticks([])
plt.xlim(0,11)
plt.yticks(fontsize=20)
plt.ylabel('ACC',fontsize=25)
plt.legend(fontsize=20,loc= 'lower left')
plt.savefig('./exp/ACC lineplot.png',dpi=300)
plt.show()
######################################################################################################################

df_mcc = pd.concat([co_model['MCC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_desc['MCC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_fp['MCC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_GCN['MCC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_EGCN['MCC'].sort_values(ascending=False).reset_index().drop(columns=['index'])],axis=1)
df_mcc.columns=col

#---------- 总体预测性能比较：小提琴图 ---------
f, ax = plt.subplots(figsize=(11.5,6))
g = sns.violinplot(data=df_mcc, palette="Set2", bw=.35, cut=2, linewidth=2)
sns.despine(left=True, bottom=True)
ax.set_ylabel('MCC',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('./exp/MCC violinplot.png',dpi=300)
plt.show()

#---------- 总体预测性能比较：折线图 ---------
lw = 3
plt.figure(figsize=(10,8))
plt.plot(df_mcc['Co-Model'],lw=lw,label='Co-Model')
plt.plot(df_mcc['Des MMT-DNN'],lw=lw,label='Des MMT-DNN')
plt.plot(df_mcc['FP MMT-DNN'],lw=lw,label='FP MMT-DNN')
plt.plot(df_mcc['MMT-EGCN'],lw=lw,label='MMT-EGCN')
plt.plot(df_mcc['MMT-GCN'],lw=lw,label='MMT-GCN')
plt.grid(False)
plt.xticks([])
plt.xlim(0,11)
plt.yticks(fontsize=20)
plt.ylabel('MCC',fontsize=25)
plt.legend(fontsize=20,loc= 'lower left')
plt.savefig('./exp/MCC lineplot.png',dpi=300)
plt.show()


######################################################################################################################

df_bac = pd.concat([co_model['BAC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_desc['BAC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_fp['BAC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_GCN['BAC'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_EGCN['BAC'].sort_values(ascending=False).reset_index().drop(columns=['index'])],axis=1)
df_bac.columns=col

#---------- 总体预测性能比较：小提琴图 ---------
f, ax = plt.subplots(figsize=(11.5,6))
g = sns.violinplot(data=df_bac, palette="Set2", bw=.35, cut=2, linewidth=2)
sns.despine(left=True, bottom=True)
ax.set_ylabel('BAC',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('./exp/BAC violinplot.png',dpi=300)
plt.show()

#---------- 总体预测性能比较：折线图 ---------
lw = 3
plt.figure(figsize=(10,8))
plt.plot(df_bac['Co-Model'],lw=lw,label='Co-Model')
plt.plot(df_bac['Des MMT-DNN'],lw=lw,label='Des MMT-DNN')
plt.plot(df_bac['FP MMT-DNN'],lw=lw,label='FP MMT-DNN')
plt.plot(df_bac['MMT-EGCN'],lw=lw,label='MMT-EGCN')
plt.plot(df_bac['MMT-GCN'],lw=lw,label='MMT-GCN')
plt.grid(False)
plt.xticks([])
plt.xlim(0,11)
plt.yticks(fontsize=20)
plt.ylabel('BAC',fontsize=25)
plt.legend(fontsize=20,loc= 'lower left')
plt.savefig('./exp/BAC lineplot.png',dpi=300)
plt.show()

######################################################################################################################

df_Precision = pd.concat([co_model['Precision'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_desc['Precision'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_fp['Precision'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_GCN['Precision'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_EGCN['Precision'].sort_values(ascending=False).reset_index().drop(columns=['index'])],axis=1)
df_Precision.columns=col

#---------- 总体预测性能比较：小提琴图 ---------
f, ax = plt.subplots(figsize=(11.5,6))
g = sns.violinplot(data=df_Precision, palette="Set2", bw=.35, cut=2, linewidth=2)
sns.despine(left=True, bottom=True)
ax.set_ylabel('Precision',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('./exp/Precision violinplot.png',dpi=300)
plt.show()

#---------- 总体预测性能比较：折线图 ---------
lw = 3
plt.figure(figsize=(10,8))
plt.plot(df_Precision['Co-Model'],lw=lw,label='Co-Model')
plt.plot(df_Precision['Des MMT-DNN'],lw=lw,label='Des MMT-DNN')
plt.plot(df_Precision['FP MMT-DNN'],lw=lw,label='FP MMT-DNN')
plt.plot(df_Precision['MMT-EGCN'],lw=lw,label='MMT-EGCN')
plt.plot(df_Precision['MMT-GCN'],lw=lw,label='MMT-GCN')
plt.grid(False)
plt.xticks([])
plt.xlim(0,11)
plt.yticks(fontsize=20)
plt.ylabel('Precision',fontsize=25)
plt.legend(fontsize=20,loc= 'lower left')
plt.savefig('./exp/Precision lineplot.png',dpi=300)
plt.show()


######################################################################################################################

df_F1 = pd.concat([co_model['F1'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_desc['F1'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_fp['F1'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_GCN['F1'].sort_values(ascending=False).reset_index().drop(columns=['index']),
                    eva_EGCN['F1'].sort_values(ascending=False).reset_index().drop(columns=['index'])],axis=1)
df_F1.columns=col

#---------- 总体预测性能比较：小提琴图 ---------
f, ax = plt.subplots(figsize=(11.5, 6))
g = sns.violinplot(data=df_F1, palette="Set2", bw=.35, cut=2, linewidth=2)
sns.despine(left=True, bottom=True)
ax.set_ylabel('F1',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('./exp/F1 violinplot.png',dpi=300)
plt.show()

#---------- 总体预测性能比较：折线图 ---------
lw = 3
plt.figure(figsize=(10,8))
plt.plot(df_F1['Co-Model'],lw=lw,label='Co-Model')
plt.plot(df_F1['Des MMT-DNN'],lw=lw,label='Des MMT-DNN')
plt.plot(df_F1['FP MMT-DNN'],lw=lw,label='FP MMT-DNN')
plt.plot(df_F1['MMT-EGCN'],lw=lw,label='MMT-EGCN')
plt.plot(df_F1['MMT-GCN'],lw=lw,label='MMT-GCN')
plt.grid(False)
plt.xticks([])
plt.xlim(0,11)
plt.yticks(fontsize=20)
plt.ylabel('F1',fontsize=25)
plt.legend(fontsize=20,loc= 'lower left')
plt.savefig('./exp/F1 lineplot.png',dpi=300)
plt.show()






EGCN = pd.read_excel("I:/4Multitask/exp/GCN_result\MMT-EGCN\model11\MTGCN_test_Tox21_statistics.xls",index_col=0)
desc = pd.read_excel("I:/4Multitask/exp/Descriptors_result\MMT-DNN\MTDNN_test_Tox21_statistics.xls",index_col=0)
fp = pd.read_excel("I:/4Multitask/exp/Fingerprints_result\MMT-DNN\MTDNN_test_Tox21_statistics.xls",index_col=0)
co_mo = pd.read_excel("I:/4Multitask/exp/0ensamble\co_model11/vote_mean_test_statistics.xls",index_col=0)
GCN = pd.read_excel("I:/4Multitask\exp\GCN_result\MMT-GCN\MTGCN_test_Tox21_statistics.xls",index_col=0)

col = ['Co-Model','Des MMT-DNN','FP MMT-DNN','MMT-GCN','MMT-EGCN']

df = pd.concat([co_mo.iloc[:-1,:]['AUC'],
                desc.iloc[:-1,:]['AUC'],
                fp.iloc[:-1,:]['AUC'],
                GCN.iloc[:-1,:]['AUC'],
                EGCN.iloc[:-1,:]['AUC']],axis=1)
df.columns = col
df.to_excel('data_sig_analysis.xlsx')

