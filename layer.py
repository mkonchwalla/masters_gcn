import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConv(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # nn.init.xavier_uniform_(self.bias)
        else:
            self.register_parameter('bias', None)


    def forward(self, x, adj):
        output = torch.mm(adj, torch.mm(x, self.weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, num_feat, num_hid, num_class, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(num_feat, num_hid)
        self.gc2 = GraphConv(num_hid, num_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.softmax(x, dim=1)


