from args import parse_args
import numpy as np

ALL_LANG = ['ar', 'en', 'es', 'fr', 'ru', 'zh']
LANG_PAIR_LIST = [['en', 'zh']]

def main():

    args = parse_args()


    # load vocabulary
    word2idx = {'<null>': 0}
    idx2word = {0: '<null>'}

    idx = 1
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
    
    # create language pair
    save_data = []

    idx = 0
    for pair in LANG_PAIR_LIST:

        with open(args.source+pair[0]+'.txt', 'r') as source:
            for line in source:
                temp = str(word2idx['<2%s>' % pair[1]])
                temp = temp + ' ' + line.strip()
                save_data.append([temp])
        source.close()
        print("Done collecting source!")

        with open(args.source+pair[1]+'.txt', 'r') as target:
            for line in target:
                line = line.strip()
                save_data[idx].append(line)
                idx += 1
        target.close()
        print("Done collecting target!")

    # json file is inconvenient for streaming
    # so convert json to csv file instead
    
    save_file = args.source + 'en_zh.txt'
    with open(save_file, 'w') as fout:
        for d in save_data:
            fout.write(','.join(d))
            fout.write('\n')
    
    fout.close()
         
if __name__ == '__main__':
    main()

