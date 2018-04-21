from args import parse_args
import numpy as np
import random
import itertools

ALL_LANG = ['ar', 'en', 'es', 'fr', 'ru', 'zh']

def main():

    args = parse_args()
        
    # segment files to overcome memory limit
    fout_list = [open(args.saveto + x + '.txt', 'w') for x in args.fout_list]
    
    pairs = [['en', 'ar'], ['en', 'es'], ['en', 'fr'], ['en', 'ru'], ['en', 'zh'], ['ar', 'en'], ['es', 'en'], ['fr', 'en'], ['ru', 'en'], ['zh', 'en']]
    
    save_data = []
    idx = 0

    for pair in pairs:

        with open('%sUNv1.0.6way.%s' % (args.source, pair[0]), 'r') as source:
            for line in source:
                temp = '<2%s>' % pair[1]
                temp = temp + ' ' + line.strip()
                save_data.append([temp])
        source.close()
        print("Done collecting source %s" % pair[0])

        with open('%sUNv1.0.6way.%s' % (args.source, pair[1]), 'r') as target:
            for line in target:
                line = line.strip()
                line = '<\t>' + line
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
  
    # shuffle data
    for fout in args.fout_list:
        lines = open(args.saveto+fout+'.txt').readlines()
        random.shuffle(lines)
        open(args.saveto+fout+'_s.txt', 'w').writelines(lines)
        print("Done shuffling %s" % fout)
    
    
    # creat train data -- simply split file1 into target / source 
    source_trn = open("/scratch/rjh347/input/source_trn.txt", "w")
    target_trn = open("/scratch/rjh347/input/target_trn.txt", "w")

    with open(args.saveto + 'files1_s.txt', 'r') as train:
        for line in train:
            line = line.split('<\t>')
            assert len(line) == 2
            source_trn.write(line[0])
            source_trn.write('\n')
            target_trn.write(line[1])
    train.close()
    source_trn.close()
    target_trn.close()

    
    source_val = open("/scratch/rjh347/input/source_val.txt", "w")
    target_val = open("/scratch/rjh347/input/target_val.txt", "w")

    # creat evaluation data -- split file2 into target / source, save with 0.2 chance          
    with open(args.saveto + 'files2_s.txt', 'r') as val:
        for line in val:
            line = line.split('<\t>')
            thresh = random.randint(0, 4)
            if thresh > 3:
                assert len(line) == 2
                source_val.write(line[0])
                source_val.write('\n')
                target_val.write(line[1])
    val.close()
    source_val.close()
    target_val.close()
if __name__ == '__main__':
    main()
