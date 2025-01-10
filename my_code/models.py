import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv,GATConv,APPNP,global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import tqdm

class GraphEncoder_old(torch.nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        #self.conv1 = GATConv(in_channels=args['input_dim'], out_channels=args['output_dim'],heads=args['heads'],concat=args['concat'],dropout=args['dropout'],edge_dim=args['edge_dim'])
        self.conv1 = RGCNConv(in_channels=args['input_dim'], out_channels=args['output_dim'],num_relations=6)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GraphEncoder(torch.nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        # 初始化 GATConv 层，并设置边特征的维度
        # self.conv1 = GATConv(in_channels=args['input_dim'], 
        #                      out_channels=args['output_dim'],
        #                      heads=args['heads'],
        #                      concat=args['concat'],
        #                      dropout=args['dropout'],
        #                      edge_dim=args['edge_dim'])  # edge_dim 传递边特征的维度
        self.conv1 = RGCNConv(in_channels=args['input_dim'], out_channels=args['output_dim'],num_relations=6)

    def forward(self, x, edge_index, edge_attr=None):
        # 如果存在 edge_attr，就传递给 GATConv 层
        if edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)  # 如果没有 edge_attr，则不使用它
        return x


class MessageEncoder(torch.nn.Module):
    def __init__(self, args):
        super(MessageEncoder, self).__init__()
        self.lin1=torch.nn.Linear(args['input_dim'],args['hidden_dim'])
        self.lin2=torch.nn.Linear(args['hidden_dim'],args['output_dim'])
    def forward(self, x):
        x = F.tanh(self.lin1(x))
        x = self.lin2(x)
        return x
    
class MessageDecoder(torch.nn.Module):
    def __init__(self, args):
        super(MessageDecoder, self).__init__()
        self.lin1=torch.nn.Linear(args['input_dim'],args['hidden_dim'])
        self.lin2=torch.nn.Linear(args['hidden_dim'],args['output_dim'])
    def forward(self, x):
        x = F.tanh(self.lin1(x))
        x = self.lin2(x)
        return x