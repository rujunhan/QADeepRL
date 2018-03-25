from args import parse_args
import numpy as np
import random

ALL_LANG = ['ar', 'en', 'es', 'fr', 'ru', 'zh']
LANG_PAIR_LIST = [['en', 'zh'], ['zh', 'en']]

def main():

    args = parse_args()


    # load vocabulary
    word2idx = {}
    idx2word = {}

    idx = 0
    with open(args.source+args.vocab) as vocab:
        for v in vocab:
            v = v.strip().split('\t')[0]
            word2idx[v] = idx
            idx2word[idx] = v
            idx += 1
    vocab.close()

    # append dictionary with <2target> symbols
    for l in ALL_LANG:
        tar = '<2%s>'%l
        word2idx[tar] = idx
        idx2word[idx] = tar
        idx += 1

    np.save(args.source+'w2i.npy', word2idx)
    np.save(args.source+'i2w.npy', idx2word)

    fout_list = [open(args.source + x + '.txt', 'w') for x in args.fout_list]
    
    save_data = []

    idx = 0
    for pair in LANG_PAIR_LIST:

        with open(args.source+pair[0]+'.txt', 'r') as source:
            for line in source:
                temp = str(word2idx['<2%s>' % pair[1]])
                temp = temp + ' ' + line.strip()
                save_data.append([temp])
        source.close()
        print("Done collecting source %s" % pair[0])

        with open(args.source+pair[1]+'.txt', 'r') as target:
            for line in target:
                line = line.strip()
                save_data[idx].append(line)
                idx += 1
        target.close()
        print("Done collecting target %s" % pair[1])
    
        for d in save_data:
            fidx = random.randint(0,len(fout_list)-1)
            fout_list[fidx].write(','.join(d))
            fout_list[fidx].write('\n')
        save_data = []
        idx = 0

    # close files to avoid conflicts
    for fout in fout_list:
        fout.close()

    for fout in args.fout_list:
        lines = open(args.source+fout+'.txt').readlines()
        random.shuffle(lines)
        open(args.source+fout+'_s.txt', 'w').writelines(lines)
        print("Done shuffling %s" % fout)
         
if __name__ == '__main__':
    main()
