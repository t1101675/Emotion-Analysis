from parse_args import args
import model
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

def process_data(dir):
    word_2_id = {}
    for _, _, files in os.walk(dir):
        for filename in files:
            if (filename == "word_2_id.json"):
                continue
            f = open(os.path.join(dir, filename), "r")
            raw = f.readlines()
            id = 1
            for item in raw:
                tempL_passage = item.strip().split("\t")[2].split(" ")
                for word in tempL_passage:
                    if word not in word_2_id:
                        word_2_id[word] = id
                        id += 1
    f = open(os.path.join(dir, "word_2_id.json"), "w")
    f.write(json.dumps(word_2_id, ensure_ascii=False))    
            
def load_word_2_id(dir):
    f = open(os.path.join(dir, "word_2_id.json"), "r")
    return json.loads(f.read())

def format_batch_rnn(data, batch_size):
    num = len(data) // batch_size
    print(num)
    passage_data = []
    target_data = []
    lengths = []
    for k in range(0, num):
        batch = data[k: k + batch_size]
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        padded_passage = pad_sequence([torch.LongTensor(item[0]).cuda() for item in batch], batch_first=True)
        passage_data.append(padded_passage)
        lengths.append(torch.LongTensor([len(item[0]) for item in batch]).cuda())
        target_data.append((torch.LongTensor([item[1] for item in batch]).cuda()))
    return passage_data, target_data, lengths

def load_data(filename, word_2_id, batch_size):
    # [{index:str, passage:[], total:int, class:[]}]
    f = open(filename, "r")
    raw = f.readlines()
    #print(raw)
    data = []
    for item in raw:
        tempL = item.strip().split("\t")
        tempL_passage = tempL[2].strip().split(" ")
        tempL_target = tempL[1].strip().split(" ")
        #total = float(tempL_target[0].split(":")[1])
        target_L = [float(x.split(":")[1])for x in tempL_target[1:]]
        target = target_L.index(max(target_L))
        passage_L = [word_2_id[word] for word in tempL_passage]
        data.append((passage_L, target))
    return data

def main():
    if (args.process_data):
        process_data(args.data_dir)
        return
    torch.cuda.set_device(args.gpu_device)
    word_2_id = load_word_2_id(args.data_dir)
    if (args.model == "rnn"):
        myModel = model.RNNModel(args, len(word_2_id), 8).cuda()
    elif (args.model == "cnn"):
        myModel = model.CNNModel(args, len(word_2_id))
    else:
        print("invalid model type")
        exit(1)

    if (args.test_only):
        test_data = load_data(os.path.join(args.data_dir, "test.in"), word_2_id, args.batch_size)
        test(myModel, word_2_id, args)
    else:
        train_data = load_data(os.path.join(args.data_dir, "train.in"), word_2_id, args.batch_size)
        valid_data = load_data(os.path.join(args.data_dir, "test.in"), word_2_id, args.batch_size)
        train_rnn(myModel, train_data, valid_data, args)
    

def train_rnn(model, train_data, valid_data, args):
    passage_data, target_data, lengths = format_batch_rnn(train_data, args.batch_size)
    #print(target_data)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #test(model, valid_data, args)
    for epoch in range(args.epoch):
        print("[!] epoch: {}".format(epoch))
        total_loss = 0
        for i in tqdm(range(len(passage_data))):
            model.zero_grad()
            score = model(passage_data[i], lengths[i])
            loss = loss_func(score, target_data[i])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("[!] Training loss: {}".format(total_loss / len(passage_data)))
        test(model, valid_data, args)

def test(model, test_data, args):
    print("[!] Start Testing:")
    passage_data, target_data, lengths = format_batch_rnn(test_data, args.batch_size)
    loss_func = nn.CrossEntropyLoss()
    test_total_loss = 0
    corret = 0
    for i in tqdm(range(len(passage_data))):
        model.zero_grad()
        score = model(passage_data[i], lengths[i])
        loss = loss_func(score, target_data[i])
        test_total_loss += loss.item()
        for j in range(args.batch_size):
            #print(torch.max(score[j], 0))
            max_index = torch.max(score[j], 0)[1].data
            if max_index == target_data[i][j]:
                corret += 1
    print("[!] Testing loss: {}, Rate: {}".format(test_total_loss / len(passage_data), corret / len(test_data)))

if __name__ == "__main__":
    main()