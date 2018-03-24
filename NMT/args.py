import argparse

def parse_args():


    parser = argparse.ArgumentParser()

    parser.add_argument('-emb', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-n_epochs', type=int, default=1)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default =0.05)
    parser.add_argument('-vocab', type=str)
    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)
    parser.add_argument('-data_file', type=str)
    parser.add_argument('-max_sent_len', type=int, default=100)
    parser.add_argument('-file_stamp', type=str)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-save_stamp', type=str)
    parser.add_argument('-best_model', type=str, default='model_best.pth.tar')
    
    parser.add_argument('-load_model', type=bool, default=False)
    args = parser.parse_args()

    args.vocab = "NMT_vocab.txt"
    args.saveto = "../results/"
    args.source = "/scratch/rjh347/data/"
    args.data_file = 'en_zh.txt'
    args.file_stamp = 'NMT'
    args.save_stamp = 'NMT_save'

    args.cuda = False
    args.best_model = 'model_best_%s.pth.tar' % (args.save_stamp)


    return args

