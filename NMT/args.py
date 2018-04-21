import argparse

def parse_args():


    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)
    parser.add_argument('-data_file', type=str)
    parser.add_argument('-max_sent_len', type=int, default=100)
    parser.add_argument('-file_stamp', type=str)
    parser.add_argument('-save_stamp', type=str)
    parser.add_argument('-fout_list', type=list, default=[])
    args = parser.parse_args()

    args.saveto = "/scratch/rjh347/input/"
    args.source = "/scratch/rjh347/data/"
    args.file_stamp = ''
    args.save_stamp = ''
    args.fout_list = ['files1', 'files2', 'files3', 'files4', 'files5']

    return args

