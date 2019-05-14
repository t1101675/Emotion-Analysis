from parse_args import args
import model
import os
import json
import torch
import torch.optim as optim
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
        padded_passage = pad_sequence([torch.FloatTensor(item[0]) for item in batch], batch_first=True)
        passage_data.append(padded_passage)
        lengths.append(torch.FloatTensor([len(item[0]) for item in batch]))
        target_data.append(torch.FloatTensor([item[1] for item in batch]))
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
        total = float(tempL_target[0].split(":")[1])
        target_L = [float(x.split(":")[1]) / total for x in tempL_target[1:]]
        passage_L = [word_2_id[word] for word in tempL_passage]
        data.append((passage_L, target_L))
    return data

def main():
    if (args.process_data):
        process_data(args.data_dir)
        return
    word_2_id = load_word_2_id(args.data_dir)
    if (args.model == "rnn"):
        myModel = model.RNNModel(args, len(word_2_id), 8)
    elif (args.model == "cnn"):
        myModel = model.CNNModel(args, len(word_2_id))
    else:
        print("invalid model type")
        exit(1)

    if (args.test_only):
        test(myModel, word_2_id, args)
    else:
        train(myModel, word_2_id, args)
    

def train(model, word_2_id, args):
    train_data = load_data(os.path.join(args.data_dir, "train.in"), word_2_id, args.batch_size)
    print(len(train_data))
    if (args.model == "rnn"):
        passage_data, target_data, lengths = format_batch_rnn(train_data, args.batch_size)
        print(lengths)

def test(model, word_2_id, args):
    test_data = load_data(os.path.join(args.data_dir, "test.in"), word_2_id, args.batch_size)
    pass

if __name__ == "__main__":
    main()