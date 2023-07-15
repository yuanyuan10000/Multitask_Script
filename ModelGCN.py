import dgl
import torch
import dgl.function as fn   #使用内置函数并行更新API
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import WeightAndSum


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


class GCNPredictor(nn.Module):
    def __init__(self, in_feats,
                 hidden_feats_tower=None,
                 gnn_norm_tower=None,
                 activation_tower=None,
                 residual_tower=None,
                 batchnorm_tower=None,
                 gcn_dropout_tower=None,
                 predictor_hidden_feats=50,
                 predictor_dropout=0.3,
                 n_tasks=2):
        super(GCNPredictor, self).__init__()

        self.gnn_tower = GCN(in_feats=in_feats,
                            hidden_feats=hidden_feats_tower,
                            gnn_norm=gnn_norm_tower,
                            activation=activation_tower,
                            residual=residual_tower,
                            batchnorm=batchnorm_tower,
                            dropout=gcn_dropout_tower)
        gnn_out_feats = self.gnn_tower.hidden_feats[-1]

        self.readout = WeightedSumAndMax(gnn_out_feats)

        self.predict = MLPPredictor(gnn_out_feats, predictor_hidden_feats,
                                           n_tasks, predictor_dropout)
    def forward(self, bg, feat):
        # 两层图卷积层

        node_feat = self.gnn_tower(bg, feat)
        graph_feat = self.readout(bg, node_feat)
        out= self.predict(graph_feat)
        # out = F.softmax(out, dim=1)
        return out

###########################################################################################################
# 以下为旧版GCN模型
'''def collate(samples):
    # 输入`samples`是一个列表# 每个元素都是一个二元组 (图, 标签)
    # 生成graoh，labels两个列表
    graphs, labels = map(list, zip(*samples))  # map函数将第二个参数（一般是数组）中的每一个项，处理为第一个参数的类型。
    # DGL提供了一个dgl.batch()方法，生成batch_graphs.
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).view(-1)

# GCN 分类器
# 先通过多层神经网络计算每个样本的表示（representation），再通过表示计算出每个类别的概率，最后通过向后传播计算梯度。

'传递节点特征h的message'
#  把一个有向边的源节点的"h"信息复制一份到目标节点的信息邮箱里的”m"区域。
# 可当作一种最简单的计算邻居带给自己的信息的方式
msg = fn.copy_src(src='h', out='m')
# # 其等价于如下：
# def message_func(edges):
#     return {'m': edges.src['h']}


'聚合邻居节点的特征,对所有邻居节点特征进行平均，并使用它来覆盖原始节点特征'
#  把邮箱中的信息进行聚合，并保存在节点的某一个特征里。
def reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


'使用ReLU（Whv + b）更新节点特征hv.'
# 对收到的消息应用线性变换和激活函数，将节点特征 hv 更新为 ReLU(Whv+b).
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}

'定义GCN'
# 我们把所有的小模块串联起来成为 GCNLayer。
# GCN实际上是对所有节点进行 消息传递/聚合/更新
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature   # 使用 h 初始化节点特征。feature应该对应的整个图的特征矩阵
        # #使用 update_all接口和自定义的消息传递及累和函数更新节点表示。
        # update_all: 1.计算所有邻居给自己带来的信息2.聚合这些信息。
        g.update_all(msg, reduce)
        # apply_nodes：  # 更新节点特征
        g.apply_nodes(func=self.apply_mod)
        # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_hidden3,dim_out):
        super(Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, dim_hidden1, F.relu)   # 每个节点特征维数为100
        self.gc2 = GCNLayer(dim_hidden1, dim_hidden2, None)   # 每个节点特征维数为20
        self.fc1 = nn.Linear(dim_hidden2, dim_hidden3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(dim_hidden3, dim_out)
        self.dropout = nn.Dropout()
    def forward(self, g):
        # 两层图卷积层
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        # dgl.mean_nodes:读出函数，将每个图的所有节点的输出特征的均值作为图的表示
        hg = dgl.mean_nodes(g, 'h')
        # 两层线性层
        out = self.dropout(F.relu(self.fc1(hg)))
        out = F.softmax(self.fc2(out), dim=1)
        return out
'''