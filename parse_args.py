import argparse
parser = argparse.ArgumentParser(description='Amotional Analysis')

parser.add_argument("--model", type=str, default="rnn", help="select model type")
parser.add_argument("--test_only", action="store_true", default=False, help="select test only mode")
parser.add_argument("--train_data", type=str, default="", help="train data file")
parser.add_argument("--valid_data", type=str, default="", help="valid data file")
parser.add_argument("--test_data", type=str, default="", help="test data file")

args = parser.parse_args()
