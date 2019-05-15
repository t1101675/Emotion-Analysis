import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

        self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.hidden_dim, dropout = self.dropout, num_layers = self.num_layers, batch_first=True)
        self.hidden_2_target = nn.Linear(self.hidden_dim, self.target_size)
        self.h = self.init_h()
        self.c = self.init_c()
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        
    def init_c(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()

    def init_h(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()

    def forward(self, passage_data, lengths):
        embeds = self.word_embeddings(passage_data)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        out, (hn, cn) = self.lstm(packed_embeds, (self.h, self.c))
        #unpacked_out, _ = pad_packed_sequence(out, batch_first=True)
        #print("hn:", hn[-1, :, :].size())
        target_space = self.hidden_2_target(hn[-1, :, :])
        #score = F.softmax(target_space, dim=1)
        return target_space

class CNNModel(nn.Module):
    def __init__(self, args, vocab_size):
        super(CNNModel, self).__init__()
        pass