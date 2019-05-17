from parse_args import args
import model
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

def main():
    data_loader = DataLoader(args)
    if (args.process_data):
        data_loader.process_data()
        return
    torch.cuda.set_device(args.gpu_device)

    data_loader.load()
    if (args.model == "rnn"):
        myModel = model.RNNModel(args, data_loader.vocab_size, 8, data_loader.id_2_vec).cuda()
    elif (args.model == "cnn"):
        myModel = model.CNNModel(args, data_loader.vocab_size, 8, data_loader.id_2_vec).cuda()
    elif (args.model == "baseline"):
        myModel = model.Baseline(args, data_loader.vocab_size, 8, data_loader.id_2_vec).cuda()
    else:
        print("invalid model type")
        exit(1)

    if (args.test_only):
        test(myModel, data_loader, args)
    else:
        train(myModel, data_loader, args)
    

def train(model, data_loader, args):
    #print(target_data)
    #passage_data, target_data, lengths = format_batch_rnn(train_data, args.batch_size, args.loss)
    passage_data, target_data, lengths = data_loader.get_data("train")
    
    if args.loss == "CEL":
        loss_func = nn.CrossEntropyLoss()
    elif args.loss == "MSE":
        loss_func = nn.MSELoss()

    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #test(model, valid_data, args)
    for epoch in range(args.epoch):
        print("[!] epoch: {}".format(epoch))
        total_loss = 0
        for i in tqdm(range(len(passage_data))):
            optimizer.zero_grad()
            score = model(passage_data[i], lengths[i])
            loss = loss_func(score, target_data[i])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("[!] Training loss: {}".format(total_loss / len(passage_data)))
        test(model, data_loader, args)

def test(model, data_loader, args):
    print("[!] Start Testing:")
    #passage_data, target_data, lengths = format_batch_rnn(test_data, args.batch_size, args.loss)
    passage_data, target_data, lengths = data_loader.get_data("test")
    if args.loss == "CEL":
        loss_func = nn.CrossEntropyLoss()
    elif args.loss == "MSE":
        loss_func = nn.MSELoss()
    test_total_loss = 0
    corret = 0
    for i in tqdm(range(len(passage_data))):
        score = model(passage_data[i], lengths[i])
        loss = loss_func(score, target_data[i])
        test_total_loss += loss.item()
        for j in range(args.batch_size):
            #print(torch.max(score[j], 0))
            max_index = torch.max(score[j], 0)[1].data
            if args.loss == "CEL":
                target_index = target_data[i][j]
            elif args.loss == "MSE":
                target_index = torch.max(target_data[i][j], 0)[1].data
            if max_index == target_index:
                corret += 1
    print("[!] Testing loss: {}, Rate: {}".format(test_total_loss / len(passage_data), corret / len(data_loader.test_data)))

if __name__ == "__main__":
    main()