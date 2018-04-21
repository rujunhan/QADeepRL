import torch

FILEPATH = '/scratch/rjh347/output/'
FILENAME = 'NMT0419.vocab.pt'


def main():

    vocab = torch.load(FILEPATH + FILENAME)
    source_vocab = vocab[0][1]
    target_vocab = vocab[1][1]
    
    src_vcb = open(FILEPATH + 'src_vcb.txt', 'w')
    tgt_vcb = open(FILEPATH + 'tgt_vcb.txt', 'w')

    for k in source_vocab.stoi.keys():
        src_vcb.write(k + '\n')
    for k in target_vocab.stoi.keys():
        tgt_vcb.write(k + '\n')


if __name__ == '__main__':
    main()
