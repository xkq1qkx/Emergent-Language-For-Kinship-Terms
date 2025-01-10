import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from sklearn.metrics import f1_score
import networkx as nx
import random
import numpy as np
from models import GraphEncoder, MessageEncoder, MessageDecoder
from quantizer import VectorQuantizer
from mydata import KinshipDataset
import sys

# 定义输出到文件的重定向
sys.stdout = open('result—1.txt', 'w')  # 打开 result.txt 文件并重定向输出

# 模型参数
graphencoder_args = {
    'input_dim': 3,
    'output_dim': 8,
    'heads': 4,
    'concat': True,
    'dropout': 0,
    'add_self_loops': True,
    'edge_dim': 3
}

messageencoder_args = {
    'input_dim': 33,
    'hidden_dim': 64,
    'output_dim': 32,
}

codebook_args = {
    'n_embeddings': 2,
    'embedding_dim': 2,
    'beta': 0.25,
}

messagedecoder_args = {
    'input_dim': 65,
    'hidden_dim': 512,
    'output_dim': 1,
}




dataset = KinshipDataset()
batch_size=3
max_nodes=4
train_loader=DataLoader(dataset,batch_size=batch_size)



# 初始化模型
graph_encoder = GraphEncoder(graphencoder_args)
message_encoder = MessageEncoder(messageencoder_args)
quantizer = VectorQuantizer(codebook_args)
message_decoder = MessageDecoder(messagedecoder_args)

# 损失函数和优化器
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for the class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, inputs, targets):
        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate cross entropy
        cross_entropy = F.cross_entropy(inputs, targets, reduction='none')

        # Compute probabilities
        probs = torch.exp(-cross_entropy)

        # Compute focal loss
        focal_loss = self.alpha * (1 - probs) ** self.gamma * cross_entropy

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
optimizer = optim.Adam(list(graph_encoder.parameters()) + 
                       list(message_encoder.parameters()) + 
                       list(quantizer.parameters()) + 
                       list(message_decoder.parameters()), lr=0.001)

# 训练函数

all_labels_epoch = []
all_predictions_epoch = []

def train():
    global all_labels_epoch, all_predictions_epoch  # 引用外部的列表
    all_labels_epoch = []  # 每个 epoch 开始时清空列表
    all_predictions_epoch = []  # 每个 epoch 开始时清空列表

    graph_encoder.train()
    message_encoder.train()
    quantizer.train()
    message_decoder.train()
    
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        
        # 获取图数据对象中的内容
        x = batch.x  # 节点特征
        edge_index = batch.edge_index  # 边索引
        edge_attr = batch.edge_attr  # 边特征
        edge_attr = edge_attr.expand(-1, 4)
        
        # 获取额外的输入数据
        caller_listener_vector = batch.caller_listener_vector
        test_listener_vector = batch.test_listener_vector
        labels = batch.labels  # 标签
        labels = labels.unsqueeze(1)  # 转换为形状 [15, 1]

        # 图编码器
        graph_emb = graph_encoder(x, edge_index)

        encoder_input = torch.cat([graph_emb, caller_listener_vector], dim=-1)

        # 消息编码器
        message = message_encoder(encoder_input)
        
        # 向量量化
        loss, quantized_message, _, _, _ = quantizer(message)
        
        # 拼接输入解码器
        decoder_input = torch.cat([graph_emb, quantized_message, test_listener_vector], dim=-1)
        
        # 消息解码器
        output = message_decoder(decoder_input)

        # 计算损失
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # 收集每个 batch 的标签和预测值
        # zero_matrix = torch.zeros_like(labels)
        all_labels_epoch.append(labels.detach().numpy())
        all_predictions_epoch.append((output.detach() > 0.5).float().numpy())
        # all_predictions_epoch.append(labels.detach().numpy())

    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss (Epoch {epoch+1}): {avg_loss}")

    # 在每个 epoch 结束后计算 F1
    compute_f1_epoch()

def compute_f1_epoch():
    # 拼接当前 epoch 的所有标签和预测值
    all_labels_flat = np.concatenate(all_labels_epoch, axis=0)
    all_predictions_flat = np.concatenate(all_predictions_epoch, axis=0)

    # 计算当前 epoch 的 F1 分数
    f1 = f1_score(all_labels_flat, all_predictions_flat)
    print(f"F1 Score (Epoch {epoch+1}): {f1:.4f}")
    return f1

# 设置训练轮次
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train()