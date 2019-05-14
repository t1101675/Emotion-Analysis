from parse_args import args
import model

def load_data(filename):
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
        item_dict["passage"] = tempL_passage
        tempL_class = tempL[1].strip().split(" ")
        item_dict["total"] = int(tempL_class[0].split(":")[1])
        item_dict["class"] = [int(x.split(":")[1]) for x in tempL_class[1:]]
        data.append(item_dict)
    return data

def main():
    if (args.test_loader):
        data = load_data(args.train_data)
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
    train_data = load_data(args.train_data)
    valid_data = load_data(args.valid_data)
    pass

def test(model, args):
    test_data = load_data(args.test_data)
    pass

if __name__ == "__main__":
    main()