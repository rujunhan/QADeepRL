import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    # Name and directories                                                                                                                                            
    parser.add_argument('-model_name', type=str, default='RL')
    parser.add_argument('-data_dir', type=str, default='/scratch/rjh347/data/squad/')
    parser.add_argument('-out_dir', type=str, default='/scratch/rjh347/data/squad/')
    parser.add_argument('-shared_path', type=str, default='/scratch/rjh347/data/squad/')
    parser.add_argument('-NMT_path', type=str, default='/scratch/rjh347/output/')
    parser.add_argument('-NMT_file', type=str, default='NMT0419_model_acc_74.72_ppl_2.96_e13.pt')
    parser.add_argument('-QA_best_model', type=str, default='model_best_lr001.pth.tar')
    parser.add_argument('-RL_path', type=str, default='/scratch/rjh347/RL/')
    parser.add_argument('-RL_file', type=str, default='input.txt')
    parser.add_argument('-RL_rewrite', type=str, default='rewrite.txt')

    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-n_episodes', type=int, default=1)

    args = parser.parse_args()

    return args
