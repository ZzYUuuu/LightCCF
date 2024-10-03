import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Run Reproduct")
    parse.add_argument('--seed', type=int, default=2023, help='random seed')
    parse.add_argument('--gpu', type=int, default=2, help='indicates which gpu to use')
    parse.add_argument('--cuda', type=bool, default=True, help='use gpu or not')
    parse.add_argument('--log', type=str, default='None', help='init log file name')
    parse.add_argument('--dataset_path', type=str, default='./dataset/', help='choice dataset')
    parse.add_argument('--dataset_type', type=str, default='.txt', help='choice dataset')
    parse.add_argument('--dataset', type=str, default='douban-book', help='choice dataset')
    parse.add_argument('--top_K', type=str, default='[10, 20, 30, 40, 50]')
    parse.add_argument('--train_epoch', type=int, default=600)
    parse.add_argument('--early_stop', type=int, default=10)
    parse.add_argument('--embedding_size', type=int, default=64)
    parse.add_argument('--train_batch_size', type=int, default=2048)
    parse.add_argument('--test_batch_size', type=int, default=2048)
    parse.add_argument('--learn_rate', type=float, default=0.001)
    parse.add_argument('--reg_lambda', type=float, default=0.0001)
    parse.add_argument('--gcn_layer', type=int, default=3)
    parse.add_argument('--test_frequency', type=int, default=1)
    parse.add_argument('--sparsity_test', type=int, default=0)
    parse.add_argument('--tau', type=float, default=0.28)
    parse.add_argument('--ssl_lambda', type=float, default=5.0)
    parse.add_argument('--encoder', type=str, default='MF')


    return parse.parse_args()