import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, args, vocab_size, target_size):
        super(RNNModel, self).__init__()
        self.batch_size = args.batch_size
        self.vocab_size = vocab_size
        self.target_size = target_size
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers

        self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.hidden_dim, dropout = self.dropout, num_layers = self.num_layers)
        self.hidden_2_target = nn.Linear(self.hidden_dim, self.target_size)
        self.hidden = self.init_hidden()
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

    def forward(self, batch):
        embeds = self.word_embeddings(batch)
        out, _ = self.lstm(embeds, self.hidden)
        target_space = self.hidden_2_target(out)
        score = F.log_softmax(target_space)
        return score

class CNNModel(nn.Module):
    def __init__(self, args, vocab_size):
        super(CNNModel, self).__init__()
        pass