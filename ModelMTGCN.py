import dgl
import torch
import dgl.function as fn   #使用内置函数并行更新API
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import WeightAndSum
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats, gnn_norm='none', activation=None,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=gnn_norm, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)


    def forward(self, g, feats):
        """ Update node representations """
        new_feats = self.graph_conv(g, feats)
        # 残差连接
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        # dropout层
        new_feats = self.dropout(new_feats)
        # 标准化层
        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class WeightedSumAndMax(nn.Module):
    """Apply weighted sum and max pooling to the node
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):

        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        # 按列拼接
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)

        return h_g


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCN, self).__init__()
        # 自定义参数
        if hidden_feats is None:
            hidden_feats = [64, 64]
        n_layers = len(hidden_feats)  # GCN层数
        self.hidden_feats = hidden_feats

        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.5 for _ in range(n_layers)]  # 每个节点以概率dropout不工作

        # 检查bug
        lengths = [len(hidden_feats), len(gnn_norm), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)  # set去重

        self.gnn_layers = nn.ModuleList()

        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], gnn_norm[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def forward(self, g, feats):
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats


class base_layer(nn.Module):
    def __init__(self, hidden1, hidden2, batchnorm='none',dropout=0.):
        super(base_layer, self).__init__()
        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU())
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(hidden2)
    def forward(self,feature):
        output = self.linear_layer(feature)
        if self.bn:
            output=self.bn_layer(output)
        return output


class MLPPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats, n_tasks, norm=None, dropout=None):
        super(MLPPredictor, self).__init__()

        n_layers = len(hidden_feats)

        if dropout is None:
            dropout = [0.5 for _ in range(n_layers)]
        if norm is None:
            norm = [True for _ in range(n_layers)]

        self.predicts = nn.ModuleList()
        for i in range(n_layers):
            self.predicts.append(base_layer(in_feats, hidden_feats[i], norm[i], dropout[i]))
            in_feats = hidden_feats[i]

        self.output_layer = nn.Linear(hidden_feats[-1], n_tasks)

    def forward(self, feats):
        for predict_layer in self.predicts:
            feats = predict_layer(feats)
        output = self.output_layer(feats)
        return output


# class MLPPredictor(nn.Module):
#
#     def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
#         super(MLPPredictor, self).__init__()
#
#         self.predict = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(in_feats, hidden_feats),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_feats),
#             nn.Linear(hidden_feats, n_tasks)
#         )
#
#     def forward(self, feats):
#         return self.predict(feats)


class MTGCNPredictor(nn.Module):
    def __init__(self, in_feats, tasks_num,
                 hidden_feats_shared=None, hidden_feats_tower=None,
                 gnn_norm_shared=None, gnn_norm_tower=None,
                 activation_shared=None, activation_tower=None,
                 residual_shared=None, residual_tower=None,
                 batchnorm_shared=None, batchnorm_tower=None,
                 gcn_dropout_shared=None, gcn_dropout_tower=None,
                 predictor_hidden_feats=None, batchnorm_predict = None, predictor_dropout=None,
                 n_tasks=2):
        super(MTGCNPredictor, self).__init__()
        # 设置默认参数
        self.gnn_shared = GCN(in_feats=in_feats,
                              hidden_feats=hidden_feats_shared,
                              gnn_norm=gnn_norm_shared,
                              activation=activation_shared,
                              residual=residual_shared,
                              batchnorm=batchnorm_shared,
                              dropout=gcn_dropout_shared)
        gnn_out_feats = self.gnn_shared.hidden_feats[-1]

        gnn_in_feats = self.gnn_shared.hidden_feats[-1]
        # self.gnn_tower = clones(GCN(in_feats=gnn_in_feats,
        #                             hidden_feats=hidden_feats_tower,
        #                             gnn_norm=gnn_norm_tower,
        #                             activation=activation_tower,
        #                             residual=residual_tower,
        #                             batchnorm=batchnorm_tower,
        #                             dropout=gcn_dropout_tower), tasks_num)
        # gnn_out_feats = self.gnn_tower[0].hidden_feats[-1]

        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                           n_tasks, batchnorm_predict,predictor_dropout), tasks_num)
    def forward(self, bg, feats,tasks_num):
        # 两层图卷积层
        outs = []
        for i in range(tasks_num):
            locals()['node_feats' + str(i)] = self.gnn_shared(bg[i], feats[i])
            # locals()['node_feats' + str(i)] = self.gnn_tower[i](bg[i], locals()['node_feats' + str(i)])
            locals()['graph_feats'+str(i)] = self.readout[i](bg[i], locals()['node_feats'+str(i)])
            locals()['out'+str(i)] = self.predict[i](locals()['graph_feats'+str(i)])
            outs.append(locals()['out'+str(i)])
        return outs


class MTEGCNPredictor(nn.Module):
    def __init__(self, in_feats, dim_self_feat, tasks_num,
                 hidden_feats_shared=None, hidden_feats_tower=None,
                 gnn_norm_shared=None, gnn_norm_tower=None,
                 activation_shared=None, activation_tower=None,
                 residual_shared=None, residual_tower=None,
                 batchnorm_shared=None, batchnorm_tower=None,
                 gcn_dropout_shared=None, gcn_dropout_tower=None,
                 predictor_hidden_feats=None, batchnorm_predict = None, predictor_dropout=None,
                 n_tasks=2):
        super(MTEGCNPredictor, self).__init__()

        self.gnn_shared = GCN(in_feats=in_feats,
                              hidden_feats=hidden_feats_shared,
                              gnn_norm=gnn_norm_shared,
                              activation=activation_shared,
                              residual=residual_shared,
                              batchnorm=batchnorm_shared,
                              dropout=gcn_dropout_shared)
        # gnn_out_feats = self.gnn_shared.hidden_feats[-1]

        gnn_in_feats = self.gnn_shared.hidden_feats[-1]
        self.gnn_tower = clones(GCN(in_feats=gnn_in_feats,
                                    hidden_feats=hidden_feats_tower,
                                    gnn_norm=gnn_norm_tower,
                                    activation=activation_tower,
                                    residual=residual_tower,
                                    batchnorm=batchnorm_tower,
                                    dropout=gcn_dropout_tower), tasks_num)
        gnn_out_feats = self.gnn_tower[0].hidden_feats[-1]

        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(2 * gnn_out_feats+dim_self_feat, predictor_hidden_feats,
                                    n_tasks, batchnorm_predict,predictor_dropout),tasks_num)
    def forward(self, bg, feats, self_feat,tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['node_feats' + str(i)] = self.gnn_shared(bg[i], feats[i])
            locals()['node_feats' + str(i)] = self.gnn_tower[i](bg[i], locals()['node_feats' + str(i)])
            locals()['graph_feats' + str(i)] = self.readout[i](bg[i], locals()['node_feats' + str(i)])
            locals()['graph_feats' + str(i)] = torch.cat((locals()['graph_feats' + str(i)], self_feat[i]), dim=1)
            locals()['out' + str(i)] = self.predict[i](locals()['graph_feats' + str(i)])
            outs.append(locals()['out' + str(i)])
        return outs


##############################################################################################################

# 以下为无GCN类的pred函数
# 即GCN layer不集成
'''
class MTGCNPredictor(nn.Module):
    def __init__(self, in_feats, tasks_num, hidden_feats=None,
                 gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                 predictor_hidden_feats=128, predictor_dropout=0., n_tasks=2):
        super(MTGCNPredictor, self).__init__()
        # 设置默认参数
        if hidden_feats is None:
            hidden_feats = [64, 64]
        n_layers = len(hidden_feats)   # GCN层数
        self.hidden_feats = hidden_feats

        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]

        # gcn层（两层GCN层的情况）
        self.gnn1 = GCNLayer(in_feats, hidden_feats[0],
                             gnn_norm[0], activation[0],residual[0],
                             batchnorm[0], dropout[0])   # 每个节点特征维数为100
        self.gnn2 = clones(GCNLayer(hidden_feats[0], hidden_feats[1],
                                    gnn_norm[1], activation[1],residual[1],
                                    batchnorm[1], dropout[1]),tasks_num)    # 每个节点特征维数为20
        # readout层
        gnn_out_feats = hidden_feats[-1]
        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                           n_tasks, predictor_dropout), tasks_num)
    def forward(self, bg, feats,tasks_num):
        # 两层图卷积层
        outs = []
        for i in range(tasks_num):
            locals()['feats'+str(i)] = self.gnn1(bg[i], feats[i])
            locals()['node_feats'+str(i)] = self.gnn2[i](bg[i], locals()['feats' + str(i)])
            locals()['graph_feats'+str(i)] = self.readout[i](bg[i], locals()['node_feats'+str(i)])
            locals()['out'+str(i)] = self.predict[i](locals()['graph_feats'+str(i)])
            outs.append(locals()['out'+str(i)])
        return outs


class MTEGCNPredictor(nn.Module):
    def __init__(self, in_feats, dim_self_feat, tasks_num, hidden_feats=None,
                 gnn_norm=None, activation=None,residual=None, batchnorm=None, dropout=None,
                 predictor_hidden_feats=128, predictor_dropout=0., n_tasks=2):
        super(MTEGCNPredictor, self).__init__()
        # 设置默认参数
        if hidden_feats is None:
            hidden_feats = [64, 64]
        n_layers = len(hidden_feats)   # GCN层数
        self.hidden_feats = hidden_feats

        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        # gcn层（两层GCN层的情况）
        self.gnn1 = GCNLayer(in_feats, hidden_feats[0], gnn_norm[0], activation[0],
                                            residual[0], batchnorm[0], dropout[0])   # 每个节点特征维数为100
        self.gnn2 = clones(GCNLayer(hidden_feats[0], hidden_feats[1], gnn_norm[1], activation[1],
                                            residual[1], batchnorm[1], dropout[1]),tasks_num)   # 每个节点特征维数为20
        # readout层
        gnn_out_feats = hidden_feats[-1]
        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(2 * gnn_out_feats+dim_self_feat, predictor_hidden_feats,
                                    n_tasks, predictor_dropout),tasks_num)
    def forward(self, bg, feats, self_feat,tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['feats' + str(i)] = self.gnn1(bg[i], feats[i])
            locals()['node_feats' + str(i)] = self.gnn2[i](bg[i], locals()['feats' + str(i)])
            locals()['graph_feats' + str(i)] = self.readout[i](bg[i], locals()['node_feats' + str(i)])
            locals()['graph_feats' + str(i)] = torch.cat((locals()['graph_feats' + str(i)], self_feat[i]), dim=1)
            locals()['out' + str(i)] = self.predict[i](locals()['graph_feats' + str(i)])
            outs.append(locals()['out' + str(i)])
        return outs

'''


## 旧版MTGCN模型

# # 多任务神经网络
# '传递节点特征h的message'
# msg = fn.copy_src(src='h', out='m')
#
# '聚合邻居节点的特征,对所有邻居节点特征进行平均，并使用它来覆盖原始节点特征。'
# def reduce(nodes):
#     accum = torch.mean(nodes.mailbox['m'], 1)
#     return {'h': accum}
#
#
# '使用ReLU（Whv + b）更新节点特征hv.'
# class NodeApplyModule(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(NodeApplyModule, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         self.activation = activation
#
#     def forward(self, node):
#         h = self.linear(node.data['h'])
#         if self.activation is not None:
#             h = self.activation(h)
#         return {'h': h}
#
#
# '定义GCN'
# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(GCNLayer, self).__init__()
#         self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
#
#     def forward(self, g, feature):
#         g.ndata['h'] = feature  # 使用 h 初始化节点特征。feature应该对应的整个图的特征矩阵
#         # #使用 update_all接口和自定义的消息传递及累和函数更新节点表示。
#         # update_all: 1.计算所有邻居给自己带来的信息2.聚合这些信息。
#         g.update_all(msg, reduce)
#         g.apply_nodes(func=self.apply_mod)  # 更新节点特征
#         return g.ndata.pop('h')
#
# class GCNNet(nn.Module):
#     def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_hidden3, dim_out, tasks_num, dropout=0.5):
#         super(GCNNet, self).__init__()
#         self.gc1 = GCNLayer(dim_in, dim_hidden1, F.relu)  # 每个节点特征维数为100
#         self.gc2 = clones(GCNLayer(dim_hidden1, dim_hidden2, F.relu), tasks_num)  # 每个节点特征维数为20
#
#         self.fc = clones(nn.Sequential(
#             nn.Linear(dim_hidden2, dim_hidden3),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim_hidden3, dim_out)), tasks_num)
#
#     def forward(self, g, tasks_num):
#         outs = []
#         for i in range(tasks_num):
#             # 两层图卷积层
#             locals()['h' + str(i + 1)] = self.gc1(g[i], g[i].ndata['feat'])
#             locals()['h' + str(i + 1)] = self.gc2[i](g[i], locals()['h' + str(i + 1)])
#
#             g[i].ndata['h'] = locals()['h' + str(i + 1)]
#             locals()['hg' + str(i + 1)] = dgl.mean_nodes(g[i], 'h')
#             # 两层线性层
#             locals()['out' + str(i + 1)] = F.softmax(self.fc[i](locals()['hg' + str(i + 1)]), dim=1)
#             outs.append(locals()['out' + str(i + 1)])
#
#         return outs
#
# #------------------------------------------------------------------------------------------------------------------------
#
# class EGCNNet(nn.Module):
#     def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_hidden3, dim_out, tasks_num, dim_self_feat, dropout=0.5):
#         super(EGCNNet, self).__init__()
#         self.gc1 = GCNLayer(dim_in, dim_hidden1, F.relu)  # 每个节点特征维数为100
#         self.gc2 = clones(GCNLayer(dim_hidden1, dim_hidden2, F.relu), tasks_num)  # 每个节点特征维数为20
#
#         self.fc = clones(nn.Sequential(
#             nn.Linear(dim_hidden2+dim_self_feat, dim_hidden3),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim_hidden3, dim_out)), tasks_num)
#
#     def forward(self, g, self_feat, tasks_num):
#         outs = []
#         for i in range(tasks_num):
#             # 两层图卷积层
#             locals()['h' + str(i)] = self.gc1(g[i], g[i].ndata['feat'])
#             locals()['h' + str(i)] = self.gc2[i](g[i], locals()['h' + str(i)])
#             g[i].ndata['h'] = locals()['h' + str(i)]
#
#             locals()['hg' + str(i)] = dgl.mean_nodes(g[i], 'h')
#             locals()['hg' + str(i)] = torch.cat((locals()['hg' + str(i)], self_feat[i]), dim =1)
#
#             # 两层线性层
#             # locals()['out' + str(i)] = self.fc[i](locals()['hg' + str(i)])
#             locals()['out' + str(i)] = F.softmax(self.fc[i](locals()['hg' + str(i)]), dim=1)
#             outs.append(locals()['out' + str(i)])
#         return outs