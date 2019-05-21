import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNModel(nn.Module):
    def __init__(self, args, vocab_size, target_size, id_2_vec=None):
        super(RNNModel, self).__init__()
        self.batch_size = args.batch_size
        self.vocab_size = vocab_size
        self.target_size = target_size
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.embedding_size = args.embedding_size
        if args.pre_vector:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(id_2_vec), freeze=not args.finetune_pv)
            #self.word_embeddings.weight.requires_grad = True
            self.hidden_dim = self.embedding_size = self.word_embeddings.weight.size()[1]
            print(self.embedding_size)
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.h = self.init_h()
        self.c = self.init_c()
        self.hidden_2_target = nn.Linear(self.hidden_dim, self.target_size)
        self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.hidden_dim, dropout = self.dropout, num_layers = self.num_layers, batch_first=True, bidirectional=True)
        
    def init_c(self):
        return torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim).cuda()

    def init_h(self):
        return torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim).cuda()

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
    def __init__(self, args, vocab_size, target_size, id_2_vec=None):
        super(CNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.target_size = target_size
        self.embedding_size = args.embedding_size
        if args.pre_vector:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(id_2_vec), freeze=not args.finetune_pv)
            #self.word_embeddings.weight.requires_grad = True
            self.embedding_size = self.word_embeddings.weight.size()[1]
            print(self.embedding_size)
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        
        kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (size, self.embedding_size)) for size in kernel_sizes])        
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(3 * 100, self.target_size)
    def forward(self, passage_data, lengths):
        embeds = self.word_embeddings(passage_data)
        x = embeds.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        target_space = self.fc(x)
        return target_space

class Baseline(nn.Module):
    def __init__(self, args, vocab_size, target_size, id_2_vec=None):
        super(Baseline, self).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_dim = args.hidden_dim
        self.target_size = target_size
        self.vocab_size = vocab_size
        if args.pre_vector:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(id_2_vec), freeze=not args.finetune_pv)
            #self.word_embeddings.weight.requires_grad = True
            self.embedding_size = self.word_embeddings.weight.size()[1]
            print(self.embedding_size)
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        self.hidden = nn.Linear(self.embedding_size, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.target_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, passage_data, lengths):
        embeds = self.word_embeddings(passage_data)
        x = embeds.unsqueeze(1)
        #print(x.size())
        x = self.hidden(embeds)
        x = F.relu(x)
        #print(x.size())
        x, _ = torch.max(x, 1)
        #print(x.size())
        target_space = self.output(x)
        #print(target_space.size())
        return target_space
        
