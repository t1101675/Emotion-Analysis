import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence

class DataLoader(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.word_2_id = {}
        self.target_type = args.loss
        self.vocab_size = 0
        self.batch_size = args.batch_size
        self.train_data = []
        self.test_data = []
        self.valid_data = []

    def process_data(self):
        for _, _, files in os.walk(self.data_dir):
            for filename in files:
                if (filename == "word_2_id.json"):
                    continue
                f = open(os.path.join(self.data_dir, filename), "r")
                raw = f.readlines()
                id = 1
                for item in raw:
                    tempL_passage = item.strip().split("\t")[2].split(" ")
                    for word in tempL_passage:
                        if word not in self.word_2_id:
                            self.word_2_id[word] = id
                            id += 1
        self.word_2_id["<pad>"] = 0
        self.vocab_size = len(self.word_2_id)
        f = open(os.path.join(self.data_dir, "word_2_id.json"), "w")
        f.write(json.dumps(self.word_2_id, ensure_ascii=False)) 

    def load_from_file(self, filename):
        f = open(filename, "r")
        raw = f.readlines()
        data = []
        #print(self.word_2_id.__class__)
        for item in raw:
            tempL = item.strip().split("\t")
            tempL_passage = tempL[2].strip().split(" ")
            tempL_target = tempL[1].strip().split(" ")
            total = float(tempL_target[0].split(":")[1])
            target_L = [float(x.split(":")[1]) / float(total) for x in tempL_target[1:]]
            if (self.target_type == "CEL"):
                target = target_L.index(max(target_L))
            elif (self.target_type == "MSE"):
                target = target_L
            passage_L = [self.word_2_id[word] for word in tempL_passage]
            data.append((passage_L, target))
        return data

    def format_batch(self, data):
        num = len(data) // self.batch_size
        passage_data = []
        target_data = []
        lengths = []
        for k in range(0, num):
            batch = data[k: k + self.batch_size]
            batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
            padded_passage = pad_sequence([torch.LongTensor(item[0]).cuda() for item in batch], batch_first=True)
            passage_data.append(padded_passage)
            lengths.append(torch.LongTensor([len(item[0]) for item in batch]).cuda())
            if self.target_type == "CEL":
                target_data.append((torch.LongTensor([item[1] for item in batch]).cuda()))
            elif self.target_type == "MSE":
                target_data.append((torch.FloatTensor([item[1] for item in batch]).cuda()))

        return passage_data, target_data, lengths

    def load(self):
        try:
            f = open(os.path.join(self.data_dir, "word_2_id.json"), "r")
        except(FileNotFoundError):
            self.process_data()
            f = open(os.path.join(self.data_dir, "word_2_id.json"), "r")

        self.word_2_id = json.loads(f.read())
        self.vocab_size = len(self.word_2_id)
        self.train_data = self.load_from_file(os.path.join(self.data_dir, "train.in"))
        self.test_data = self.load_from_file(os.path.join(self.data_dir, "test.in"))
        self.valid_data = self.load_from_file(os.path.join(self.data_dir, "test.in"))
        pass

    def get_data(self, data_type):
        if data_type == "train":
            return self.format_batch(self.train_data)
        elif data_type == "test":
            return self.format_batch(self.test_data)
        else:
            return self.format_batch(self.valid_data)
        

    