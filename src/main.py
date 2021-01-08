import argparse
from data_loader import load_data
from train import train
from inference import inference

parser = argparse.ArgumentParser()
# parser.add_argument('--train_file', type=str, default='E:\\dataset\\three months simplify\\temp.json', help='path to the training file')
# parser.add_argument('--test_file', type=str, default='E:\\dataset\\three months simplify\\20170101.json', help='path to the test file')
parser.add_argument('--file_directory', type=str, default='../data/four_weeks/', help='path to the files')
parser.add_argument('--first_train_file', type=int, default=20170101, help='first train file name')
parser.add_argument('--train_file_num', type=int, default=1, help='number of train_file')
parser.add_argument('--first_test_file', type=int, default=20170101, help='first test file name')
parser.add_argument('--test_file_num', type=int, default=1, help='number of test_file')

parser.add_argument('--max_click_history', type=int, default=30, help='number of sampled click history for each user')
parser.add_argument('--n_filters', type=int, default=100, help='number of filters for each size in CNN')
parser.add_argument('--filter_sizes',  type=int, default=[3,3], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 3,3')
parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')
parser.add_argument('--time_weight', type=float, default=5, help='weight of time loss')
parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=400, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--word_dim', type=int, default=100,
                    help='dimension of word embeddings, please ensure that the specified input file exists')
parser.add_argument('--max_title_length', type=int, default=10,
                    help='maximum length of news titles, should be in accordance with the input datasets')
parser.add_argument('--GPU_DEVICE', type=str, default='1', help='GPU device to use')
parser.add_argument('--is_train', type=bool, default=True,
                            help='whether training')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='rate of dropout')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for precision, recall,f1-score')
args = parser.parse_args()

# train_data=load_data(args)
# test_data=load_data(args)

print(args)
train(args)
