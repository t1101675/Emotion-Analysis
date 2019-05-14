import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()
        self.class_dim = args.class_dim
        self.batch_size = args.batch_size
        pass

    def forward(self, batch):
        
        return 

class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()
        pass