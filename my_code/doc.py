import torch
import torch.nn as nn
import egg.core as core

from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F

import random
import numpy as np

class Vision(nn.Module):
    def __init__(self):
        super(Vision, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return x

class PretrainNet(nn.Module):
    def __init__(self, vision_module):
        super(PretrainNet, self).__init__()
        self.vision_module = vision_module
        self.fc = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc(F.leaky_relu(x))
        return x

class Sender(nn.Module):
    def __init__(self, vision, output_size):
        super(Sender, self).__init__()
        self.fc = nn.Linear(500, output_size)
        self.vision = vision
        
    def forward(self, x, aux_input=None):
        with torch.no_grad():
            x = self.vision(x)
        x = self.fc(x)
        return x
    
    
class Receiver(nn.Module):
    def __init__(self, input_size):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_size, 784)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc(channel_input)
        return torch.sigmoid(x)
    
def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    loss = F.binary_cross_entropy(receiver_output, sender_input.view(-1, 784), reduction='none').mean(dim=1)
    return loss, {}
def main():
    opts = core.init(params=['--random_seed=7', # will initialize numpy, torch, and python RNGs
                         '--lr=1e-3',   # sets the learning rate for the selected optimizer 
                         '--batch_size=32',
                         '--optimizer=adam'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.ToTensor()

    batch_size = 4 # set via the CL arguments above
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
           transform=transform),
           batch_size=batch_size, shuffle=True, **kwargs)
    #train_loader=train_loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
           batch_size=batch_size, shuffle=False, **kwargs)
    vision = Vision()
    class_prediction = PretrainNet(vision) #  note that we pass vision - which we want to pretrain
    class_prediction = class_prediction.to(device)
    
    # sender = Sender(vision, output_size=400)
    # receiver = Receiver(input_size=400)
    vocab_size = 10
    # sender = Sender(vision, vocab_size)
    # sender = core.GumbelSoftmaxWrapper(sender, temperature=1.0) # wrapping into a GS interface, requires GS temperature
    # receiver = Receiver(input_size=400)
    # receiver = core.SymbolReceiverWrapper(receiver, vocab_size, agent_input_size=400)
    # game = core.SymbolGameGS(sender, receiver, loss)

    hidden_size = 20
    emb_size = 30
    sender = Sender(vision, hidden_size)
    receiver = Receiver(hidden_size)
    sender_rnn = core.RnnSenderGS(sender, vocab_size, emb_size, hidden_size,
                                   cell="gru", max_len=2, temperature=1.0)
    receiver_rnn = core.RnnReceiverGS(receiver, vocab_size, emb_size,
                    hidden_size, cell="gru")

    game = core.SenderReceiverRnnGS(sender_rnn, receiver_rnn, loss)
    optimizer = torch.optim.Adam(game.parameters())

    trainer = core.Trainer(
        game=game, optimizer=optimizer, train_data=train_loader,
        validation_data=test_loader
    )
    n_epochs = 15
    print("starting training")
    trainer.train(n_epochs)
if __name__ == '__main__':
    main()