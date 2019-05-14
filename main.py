from parse_args import args
import model
import os
import json

def process_data(dir):
    word_2_id = {}
    for _, _, files in os.walk(dir):
        for filename in files:
            if (filename == "word_2_id.json"):
                continue
            f = open(os.path.join(dir, filename), "r")
            raw = f.readlines()
            id = 0
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

def load_data(filename, word_2_id):
    # [{index:str, passage:[], total:int, class:[]}]
    f = open(filename, "r")
    raw = f.readlines()
    #print(raw)
    data = []
    for item in raw:
        item_dict = {}
        tempL = item.strip().split("\t")
        item_dict["index"] = tempL[0]
        tempL_passage = tempL[2].strip().split(" ")
        item_dict["passage"] = [word_2_id[word] for word in tempL_passage]
        tempL_class = tempL[1].strip().split(" ")
        item_dict["total"] = int(tempL_class[0].split(":")[1])
        item_dict["class"] = [int(x.split(":")[1]) for x in tempL_class[1:]]
        data.append(item_dict)
    return data

def main():
    if (args.test_loader):
        word_2_id = load_word_2_id(args.data_dir)
        data = load_data(os.path.join(args.data_dir, "train.in"), word_2_id)
        print(data)
        return


    if (args.model == "rnn"):
        myModel = model.RNNModel(args)
    elif (args.model == "cnn"):
        myModel = model.CNNModel(args)
    else:
        print("invalid model type")
        exit(1)

    if (args.test_only):
        test(myModel, args)
    else:
        train(myModel, args)
    

def train(model, args):
    word_2_id = load_word_2_id(args.data_dir)
    train_data = load_data(os.path.join(args.data_dir, "train.in"), word_2_id)
    pass

def test(model, args):
    word_2_id = load_word_2_id(args.data_dir)
    test_data = load_data(os.path.join(args.data_dir, "test.in"), word_2_id)
    pass

if __name__ == "__main__":
    main()