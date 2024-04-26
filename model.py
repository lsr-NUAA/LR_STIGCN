import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from thop import profile


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.xavier_uniform_(self.weight, gain=math.sqrt(2.0))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        # self.bn1 = nn.BatchNorm1d(nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


#
# x = torch.randn(20, 450,39).cuda()
# adj = torch.randn(20, 450, 450).cuda()
# model = GCN(39, 10).cuda()
# out = model(x, adj)
# print(out.shape)

class tensor_GCN(nn.Module):
    def __init__(self, nfeat, nhid1, num_windows):
        super(tensor_GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.num_windows = num_windows
        self.intra_Mutil_GCN = nn.ModuleList([GCN(self.nfeat, self.nhid1) for i in range(self.num_windows)])
        self.inter_GCN = GCN(self.nfeat, self.nhid1)

    def forward(self, x, adj, tg):
        # print(x.shape)
        # 图内传播
        outs = []
        for i, gcn_layer in enumerate(self.intra_Mutil_GCN):
            features = x[:, i, :, :]
            adj_new = adj[:, i, :, :]
            out = gcn_layer(features, adj_new)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        outs = F.normalize(outs)
        outs = torch.reshape(outs, (outs.shape[0], self.num_windows * 90, -1))
        x_fea = torch.reshape(x, (x.shape[0], self.num_windows * 90, -1))  # torch.Size([20, 450, 39])
        out_inter = F.normalize(self.inter_GCN(x_fea, tg))
        outss = out_inter + outs
        return outss  # 20,450,20


# x = torch.randn(20, 5, 90, 39)
# dynet = torch.randn(20, 5, 90, 90)
# tg = torch.randn(20, 450, 450)
# model = tensor_GCN(39, 20, 5)
# out = model(x, dynet, tg)
# print(out.shape)

class hyper_graphCN(nn.Module):
    def __init__(self, nhid1, he_nhid):
        super(hyper_graphCN, self).__init__()
        self.nhid1 = nhid1
        self.he_nhid = he_nhid
        self.W = nn.Parameter(torch.Tensor(self.nhid1, self.he_nhid))
        init.xavier_uniform_(self.W, gain=math.sqrt(2.0))
        self.U = nn.Parameter(torch.Tensor(self.he_nhid, self.he_nhid))
        init.xavier_uniform_(self.U, gain=math.sqrt(2.0))

    def forward(self, H):
        incidence_matrix = torch.matmul(H, self.W)
        incidence_matrix_T = incidence_matrix.transpose(2, 1)  # torch.Size([20, 32, 450])
        emb = torch.matmul(self.U, incidence_matrix_T)  # emb torch.Size([20, 32, 450])
        hyperedge_emb_E = F.relu(torch.matmul(emb, H)) + torch.matmul(incidence_matrix_T, H)
        node_emb_F = torch.matmul(incidence_matrix, hyperedge_emb_E)
        return node_emb_F, incidence_matrix


# H = torch.randn(20,450,20)
# model = hyper_graphCN(20,32)
# out = model(H)
# print(out.shape)

class model_all(nn.Module):
    def __init__(self, nfeat, nhid1, num_windows, he_nhid):
        super(model_all, self).__init__()
        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.num_windows = num_windows
        self.he_nhid = he_nhid
        self.tensorgcn = tensor_GCN(self.nfeat, self.nhid1, self.num_windows)
        self.hypergraphCN = hyper_graphCN(self.nhid1, self.he_nhid)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(self.num_windows * 90 * self.nhid1, 90)
        self.bn1 = nn.BatchNorm1d(90)
        self.d1 = nn.Dropout(p=0.6)
        self.l2 = nn.Linear(90, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.d2 = nn.Dropout(p=0.6)
        self.l3 = nn.Linear(32, 2)
        self.logs = nn.LogSoftmax(dim=1)

    def forward(self, x, adj, tg):
        H = self.tensorgcn(x, adj, tg)  # torch.Size([2, 360, 20])
        out, incidence_matrix = self.hypergraphCN(H)
        out = self.f1(out)
        # print(out.shape)
        out = self.d1(self.bn1(self.l1(out)))
        out = self.d2(self.bn2(self.l2(out)))
        out = self.logs(self.l3(out))
        return out


x = torch.randn(2, 5, 90, 39)
dynet = torch.randn(2, 5, 90, 90)
tg = torch.randn(2, 450, 450)
model = model_all(39, 20, 5, 32)
out = model(x, dynet, tg)
print(out.shape)

flops, params = profile(model, (x, dynet, tg,))
print('flops: ', flops, 'params: ', params)
print('flops: %.3f M, params: %.3f M' % (flops / 2000000.0, params / 1000000.0))
