import argparse
parser = argparse.ArgumentParser(description='Amotional Analysis')

parser.add_argument("--model", type=str, default="rnn", help="select model type")
parser.add_argument("--test_only", action="store_true", default=False, help="select test only mode")
parser.add_argument("--data_dir", type=str, default="", help="data_dir")
parser.add_argument("--test_loader", action="store_true", default=False)
parser.add_argument("--class_dim", type=int, default=8, help="the number of classes")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
args = parser.parse_args()
