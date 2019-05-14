import argparse
parser = argparse.ArgumentParser(description='Amotional Analysis')

parser.add_argument("--model", type=str, default="rnn", help="select model type")
parser.add_argument("--test_only", action="store_true", default=False, help="select test only mode")
parser.add_argument("--data_dir", type=str, default="", help="data_dir")
parser.add_argument("--process_data", action="store_true", default=False)
parser.add_argument("--class_dim", type=int, default=8, help="the number of classes")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension for rnn")
parser.add_argument("--embedding_size", type=int, default=200, help="embedding size for word vector")
parser.add_argument("--num_layers", type=int, default=1, help="rnn layers")
parser.add_argument("--dropout", type=float, default=0.0, help="drop out")
args = parser.parse_args()
