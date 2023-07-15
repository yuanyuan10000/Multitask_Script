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
        g = dgl.add_self_loop(g)  # bigraph需要加
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
            # h_g_max = dgl.max_nodes(bg, 'h')
            # h_g_mean = dgl.mean_nodes(bg, 'h')
            # h_g_sum = dgl.sum_nodes(bg, 'h')
        # # 按列拼接
        # h_g = torch.cat([h_g_sum,h_g_max, h_g_mean], dim=1)

        return h_g_sum



class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCN, self).__init__()
        # 自定义参数
        if hidden_feats is None:
            hidden_feats = [100, 50]
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
            dropout = [0. for _ in range(n_layers)]  # 每个节点以概率dropout不工作

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



class MLPPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        return self.predict(feats)


class MTGCNPredictor(nn.Module):
    def __init__(self, in_feats, tasks_num,
                 hidden_feats_shared=None, hidden_feats_tower=None,
                 gnn_norm_shared=None, gnn_norm_tower=None,
                 activation_shared=None, activation_tower=None,
                 residual_shared=None, residual_tower=None,
                 batchnorm_shared=None, batchnorm_tower=None,
                 gcn_dropout_shared=None, gcn_dropout_tower=None,
                 predictor_hidden_feats=50, predictor_dropout=0.3,
                 n_tasks=2):
        super(MTGCNPredictor, self).__init__()
        # 设置默认参数
        # self.gnn_shared = GCN(in_feats=in_feats,
        #                       hidden_feats=hidden_feats_shared,
        #                       gnn_norm=gnn_norm_shared,
        #                       activation=activation_shared,
        #                       residual=residual_shared,
        #                       batchnorm=batchnorm_shared,
        #                       dropout=gcn_dropout_shared)
        # gnn_out_feats = self.gnn_shared.hidden_feats[-1]

        # gnn_in_feats = self.gnn_shared.hidden_feats[-1]
        gnn_in_feats = in_feats
        self.gnn_tower = clones(GCN(in_feats=gnn_in_feats,
                                    hidden_feats=hidden_feats_tower,
                                    gnn_norm=gnn_norm_tower,
                                    activation=activation_tower,
                                    residual=residual_tower,
                                    batchnorm=batchnorm_tower,
                                    dropout=gcn_dropout_tower), tasks_num)
        gnn_out_feats = self.gnn_tower[0].hidden_feats[-1]

        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(gnn_out_feats, predictor_hidden_feats,
                                           n_tasks, predictor_dropout), tasks_num)
    def forward(self, bg, feats,tasks_num):
        # 两层图卷积层
        outs = []
        for i in range(tasks_num):
            # locals()['node_feats' + str(i)] = self.gnn_shared(bg[i], feats[i])
            locals()['node_feats' + str(i)] = self.gnn_tower[i](bg[i], feats[i])
            locals()['graph_feats'+str(i)] = self.readout[i](bg[i], locals()['node_feats'+str(i)])
            locals()['out'+str(i)] = self.predict[i](locals()['graph_feats'+str(i)])
            outs.append(locals()['out'+str(i)])
        return outs


class MTEGCNPredictor(nn.Module):
    def __init__(self, in_feats, dim_self_feat, tasks_num, self_feat_hidden=32,
                 hidden_feats_shared=None, hidden_feats_tower=None,
                 gnn_norm_shared=None, gnn_norm_tower=None,
                 activation_shared=None, activation_tower=None,
                 residual_shared=None, residual_tower=None,
                 batchnorm_shared=None, batchnorm_tower=None,
                 gcn_dropout_shared=None, gcn_dropout_tower=None,
                 predictor_hidden_feats=50, predictor_dropout=0.3,
                 n_tasks=2):
        super(MTEGCNPredictor, self).__init__()
        # 设置默认参数
        self.gnn_shared = GCN(in_feats=in_feats,
                              hidden_feats=hidden_feats_shared,
                              gnn_norm=gnn_norm_shared,
                              activation=activation_shared,
                              residual=residual_shared,
                              batchnorm=batchnorm_shared,
                              dropout=gcn_dropout_shared)
        gnn_out_feats = self.gnn_shared.hidden_feats[-1]

        # # gnn_in_feats = self.gnn_shared.hidden_feats[-1]
        # gnn_in_feats = in_feats
        # self.gnn_tower = clones(GCN(in_feats=gnn_in_feats,
        #                             hidden_feats=hidden_feats_tower,
        #                             gnn_norm=gnn_norm_tower,
        #                             activation=activation_tower,
        #                             residual=residual_tower,
        #                             batchnorm=batchnorm_tower,
        #                             dropout=gcn_dropout_tower), tasks_num)
        # gnn_out_feats = self.gnn_tower[0].hidden_feats[-1]

        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(gnn_out_feats+self_feat_hidden, predictor_hidden_feats,
                                           n_tasks, predictor_dropout), tasks_num)
        self.self_feat_layer = nn.Sequential(
            nn.Linear(dim_self_feat, self_feat_hidden),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.BatchNorm1d(self_feat_hidden))

    def forward(self, bg, feats,self_feat,tasks_num):
        # 两层图卷积层
        outs = []
        for i in range(tasks_num):
            locals()['node_feats' + str(i)] = self.gnn_shared(bg[i], feats[i])
            # locals()['node_feats' + str(i)] = self.gnn_tower[i](bg[i], feats[i])
            locals()['graph_feats'+str(i)] = self.readout[i](bg[i], locals()['node_feats'+str(i)])
            locals()['self_feat'+str(i)] = self.self_feat_layer(self_feat[i])
            locals()['graph_feats' + str(i)] = torch.cat((locals()['graph_feats' + str(i)], locals()['self_feat'+str(i)]), dim=1)
            locals()['out'+str(i)] = self.predict[i](locals()['graph_feats'+str(i)])
            outs.append(locals()['out'+str(i)])
        return outs


class MTEGCNPredictor_2(nn.Module):
    def __init__(self, in_feats, dim_self_feat, tasks_num,
                 hidden_feats_shared=None, hidden_feats_tower=None,
                 gnn_norm_shared=None, gnn_norm_tower=None,
                 activation_shared=None, activation_tower=None,
                 residual_shared=None, residual_tower=None,
                 batchnorm_shared=None, batchnorm_tower=None,
                 gcn_dropout_shared=None, gcn_dropout_tower=None,
                 predictor_hidden_feats=50, predictor_dropout=0.,
                 n_tasks=2):
        super(MTEGCNPredictor_2, self).__init__()
        # 设置默认参数
        self.gnn_shared = GCN(in_feats=in_feats,
                              hidden_feats=hidden_feats_shared,
                              gnn_norm=gnn_norm_shared,
                              activation=activation_shared,
                              residual=residual_shared,
                              batchnorm=batchnorm_shared,
                              dropout=gcn_dropout_shared)
        gnn_out_feats = self.gnn_shared.hidden_feats[-1]

        # # gnn_in_feats = self.gnn_shared.hidden_feats[-1]
        # gnn_in_feats = in_feats
        # self.gnn_tower = clones(GCN(in_feats=gnn_in_feats,
        #                             hidden_feats=hidden_feats_tower,
        #                             gnn_norm=gnn_norm_tower,
        #                             activation=activation_tower,
        #                             residual=residual_tower,
        #                             batchnorm=batchnorm_tower,
        #                             dropout=gcn_dropout_tower), tasks_num)
        # gnn_out_feats = self.gnn_tower[0].hidden_feats[-1]

        self.readout = clones(WeightedSumAndMax(gnn_out_feats), tasks_num)

        self.predict = clones(MLPPredictor(gnn_out_feats+dim_self_feat, predictor_hidden_feats,
                                           n_tasks, predictor_dropout), tasks_num)

    def forward(self, bg, feats,self_feat,tasks_num):
        # 两层图卷积层
        outs = []
        for i in range(tasks_num):
            locals()['node_feats' + str(i)] = self.gnn_shared(bg[i], feats[i])
            # locals()['node_feats' + str(i)] = self.gnn_tower[i](bg[i], feats[i])
            locals()['graph_feats'+str(i)] = self.readout[i](bg[i], locals()['node_feats'+str(i)])
            locals()['graph_feats' + str(i)] = torch.cat((locals()['graph_feats' + str(i)], self_feat[i]), dim=1)
            locals()['out'+str(i)] = self.predict[i](locals()['graph_feats'+str(i)])
            outs.append(locals()['out'+str(i)])
        return outs

