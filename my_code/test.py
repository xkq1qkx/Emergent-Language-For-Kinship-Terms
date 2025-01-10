import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

#from torch.utils.data import DataLoader
from torch_geometric.utils import from_networkx
from sklearn.metrics import f1_score
import networkx as nx
import random
import numpy as np
from models import GraphEncoder, MessageEncoder, MessageDecoder
from mydata import KinshipDataset
import sys
import egg.core as core

# 定义输出到文件的重定向
sys.stdout = open('result-1.txt', 'w')  # 打开 result.txt 文件并重定向输出

# 模型参数
graphencoder_args = {
    'input_dim': 2, # 点特征维度
    'output_dim': 8,
}

messageencoder_args = {
    'input_dim': 198, # max_nodes*(graphencoder_args['output_dim']+1)=22*(8+1)// +1 是send_receive_vector
    'hidden_dim': 256,
    'output_dim': 32,
}

messagedecoder_args = {
    'input_dim': 230,# max_nodes*(graphencoder_args['output_dim']+1)+messageencoder_args['output_dim']=22*(8+1)+32
    'hidden_dim': 256,
    'output_dim': 22, # max_nodes
}

egg_args={
    'vocab_size': 3, # 0 是终止符，所以实际词典是比这个参数小1的
    'hidden_size': 32, # should == messageencoder_args['output_dim']
    'emb_size':30, 
    'max_len': 5, 
    'cell': 'gru',#lstm, gru
    'temperature': 1.0
}

focal_loss_args={
    'alpha': 0.75,
    'gamma': 2.0,
    'reduction': 'mean'
}
batch_size=32
opts = core.init(params=['--random_seed=7', # will initialize numpy, torch, and python RNGs
                         '--lr=0.01',   # sets the learning rate for the selected optimizer 
                         '--optimizer=adam'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = KinshipDataset(mode='train',max_nodes=messagedecoder_args['output_dim'])
test_dataset = KinshipDataset(mode='test',max_nodes=messagedecoder_args['output_dim'])
val_dataset = KinshipDataset(mode='val',max_nodes=messagedecoder_args['output_dim'])
train_loader=DataLoader(train_dataset,batch_size=batch_size)
test_loader=DataLoader(test_dataset,batch_size=batch_size)
val_loader=DataLoader(val_dataset,batch_size=batch_size)
count=0
# for batch in train_loader:
#     count+=1
graph_encoder=GraphEncoder(graphencoder_args)
class Sender(nn.Module):
    def __init__(self):
        super(Sender, self).__init__()
        self.graph_encoder=graph_encoder
        self.message_encoder=MessageEncoder(messageencoder_args)
        
    def forward(self, x, aux_input=None):
        
        graph_emb = self.graph_encoder(aux_input[0], aux_input[1], aux_input[2])
        output = self.message_encoder(torch.cat((graph_emb.reshape(batch_size,-1),x.reshape(batch_size,-1)),dim=1))
        return output
    
    
class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()
        self.graph_encoder=graph_encoder
        self.message_decoder=MessageDecoder(messagedecoder_args)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        graph_emb = self.graph_encoder(aux_input[0], aux_input[1], aux_input[2])
        output = self.message_decoder(torch.cat((graph_emb.reshape(batch_size,-1),channel_input.reshape(batch_size,-1),receiver_input.reshape(batch_size,-1)),dim=1))
        return torch.sigmoid(output)
    

# 损失函数和优化器


class FocalLoss(nn.Module):
    def __init__(self, alpha=focal_loss_args['alpha'], gamma=focal_loss_args['gamma'], reduction=focal_loss_args['reduction']):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for the class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, inputs, targets):
        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
         # Compute the binary cross entropy loss with logits
        # This combines a sigmoid layer and the BCE loss in one single class
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Apply sigmoid to get probabilities
        #probs = torch.sigmoid(inputs)
        probs = inputs
        # Select the probabilities of the true class
        # For targets==1, probs_t = probs
        # For targets==0, probs_t = 1 - probs
        probs_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute the modulating factor (1 - p_t)^gamma
        focal_weight = (1 - probs_t) ** self.gamma

        # Compute the alpha weighting factor
        # For targets==1, alpha_t = alpha
        # For targets==0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute the focal loss
        loss = alpha_t * focal_weight * BCE_loss

        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        '''
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
        '''
criterion = FocalLoss(alpha=focal_loss_args['alpha'], gamma=focal_loss_args['gamma'], reduction=focal_loss_args['reduction'])

def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    loss=criterion(receiver_output,_labels.reshape((batch_size,-1)))
    return loss, {}
sender = Sender()
receiver = Receiver()
sender_rnn = core.RnnSenderGS(sender, egg_args['vocab_size'], egg_args['emb_size'], egg_args['hidden_size'],
                                   cell=egg_args['cell'], max_len=egg_args['max_len'], temperature=egg_args['temperature'])
receiver_rnn = core.RnnReceiverGS(receiver, egg_args['vocab_size'], egg_args['emb_size'],
                    egg_args['hidden_size'], cell=egg_args['cell'])

game = core.SenderReceiverRnnGS(sender_rnn, receiver_rnn, loss)
#game=torch.load("../check_point/aliyun_our_method_best.pt")
optimizer = torch.optim.Adam(game.parameters(), lr = 0.001)

trainer = core.Trainer(
        game=game, optimizer=optimizer, 
        train_data=train_loader,test_data=test_loader,
        val_data=val_loader,device=device
    )

print("| Epoch | Train Loss | Train F1_score | Train Acc_score | Train Precision | Train recall | Test Loss  | Test F1_score | Test Acc_score| Test Precision | Test Recall |")
n_epochs=50
trainer.train(n_epochs)
