from parse_args import args
import model

def load_data(filename):
    # [{passgae:[], class:[]}]
    return []

def main():
    if (args.model == "rnn"):
        myModel = model.RNNModel(args)
    elif (args.model == "cnn"):
        myModel = model.CNNModel(args)
    else:
        print("invalid model type")
        exit(1)

    if (args.test_only):
        test(args)
    else:
        train(args)
    pass

def train(args):
    train_data = load_data(args.train_data)
    valid_data = load_data(args.valid_data)
    pass

def test(args):
    test_data = load_data(args.test_data)
    pass

if __name__ == "__main__":
    main()