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
        self.pre_vector = args.pre_vector
        self.id_2_vec = []
        self.vec_len = 0
        self.fix_length = args.fix_length

    def process_data(self):
        if self.pre_vector:
            f_vec = open(os.path.join(self.data_dir, self.pre_vector), "r")
            vec_L = f_vec.readlines()
            vec_dict = {}
            self.vec_len = int(vec_L[0].split(" ")[1])
            for i in range(1, len(vec_L)):
                tempL = vec_L[i].strip().split(" ")
                vec_dict[tempL[0]] = [float(x) for x in tempL[1:]]
       
        self.word_2_id["<pad>"] = 0
        self.id_2_vec = [[0 for i in range(self.vec_len)]]
        for _, _, files in os.walk(self.data_dir):
            for filename in files:
                file_type = filename.split(".")[-1]
                if (file_type not in ["in"]):
                    continue
                f = open(os.path.join(self.data_dir, filename), "r")
                raw = f.readlines()
                id = 1
                for item in raw:
                    tempL_passage = item.strip().split("\t")[2].split(" ")
                    for word in tempL_passage:
                        if word not in self.word_2_id:
                            self.word_2_id[word] = id
                            if self.pre_vector:
                                try:
                                    self.id_2_vec.append(vec_dict[word])
                                except(KeyError):
                                    self.id_2_vec.append([0 for i in range(self.vec_len)])
                            id += 1
        
        self.vocab_size = len(self.word_2_id)
        f = open(os.path.join(self.data_dir, "word_2_id.json"), "w")
        f.write(json.dumps(self.word_2_id, ensure_ascii=False)) 
        if  self.pre_vector:   
            f_id2Vec = open(os.path.join(self.data_dir, self.pre_vector + ".vec.json"), "w")
            f_id2Vec.write(json.dumps(self.id_2_vec))


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
            if self.target_type == "CEL":
                target = target_L.index(max(target_L))
            elif self.target_type == "MSE":
                target = target_L
            passage_L = [self.word_2_id[word] for word in tempL_passage]
            data.append((passage_L, target))
        return data

    def format_batch(self, data):
        num = len(data) // self.batch_size
        passage_data = []
        target_data = []
        lengths = []
        data_sorted = sorted(data, key=lambda item: len(item[0]), reverse=True)
        data_padded = pad_sequence([torch.LongTensor(item[0]).cuda() for item in data], batch_first=True)
        if self.fix_length:
            data_padded = [item[0:self.fix_length] for item in data_padded]
        for k in range(0, num):
            batch = data_sorted[k: k + self.batch_size]
            #batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
            #padded_passage = pad_sequence([torch.LongTensor(item[0]).cuda() for item in batch], batch_first=True)
            passage_data.append(data_padded[k: k + self.batch_size])
            lengths.append(torch.LongTensor([len(item[0]) for item in batch]).cuda())
            if self.target_type == "CEL":
                target_data.append((torch.LongTensor([item[1] for item in batch]).cuda()))
            else:
                target_data.append((torch.FloatTensor([item[2] for item in batch]).cuda()))

        return passage_data, target_data, lengths

    def load(self):
        try:
            f = open(os.path.join(self.data_dir, "word_2_id.json"), "r")
        except(FileNotFoundError):
            self.process_data()
            f = open(os.path.join(self.data_dir, "word_2_id.json"), "r")
        print("[!] Loading word to index ", end="")
        self.word_2_id = json.loads(f.read())
        print("Done!")
        self.vocab_size = len(self.word_2_id)
        print("[!] Loading train data ", end="")
        self.train_data = self.load_from_file(os.path.join(self.data_dir, "train.in"))
        print("Done!")
        print("[!] Loading test data ", end="")
        self.test_data = self.load_from_file(os.path.join(self.data_dir, "test.in"))
        print("Done!")       
        print("[!] Loading valid data ", end="")
        self.valid_data = self.load_from_file(os.path.join(self.data_dir, "test.in"))
        print("Done!")
        if (self.pre_vector):
            print("[!] Loading pretrained embeding ", end="")
            f = open(os.path.join(self.data_dir, self.pre_vector + ".vec.json"), "r")
            self.id_2_vec = json.loads(f.read())
            print("Done!")

    def get_data(self, data_type):
        if data_type == "train":
            return self.format_batch(self.train_data)
        elif data_type == "test":
            return self.format_batch(self.test_data)
        else:
            return self.format_batch(self.valid_data)
        

    