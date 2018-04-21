from args import parse_args
import numpy as np
import random

def main():

    args = parse_args()

    source_trn = open("/scratch/rjh347/input/source_trn.txt", "w")
    target_trn = open("/scratch/rjh347/input/target_trn.txt", "w")
    
    # creat train data -- simply split file1 into target / source
    with open(args.saveto + 'file1_s.txt', 'r') as train:
        for line in train:
            src, tgt = line.split(',')
            source_trn.write(src)
            target_trn.write(tgt)
    train.close()
    source_trn.close()
    target_trn.close()

    source_val = open("/scratch/rjh347/input/source_val.txt", "w")
    target_val = open("/scratch/rjh347/input/target_val.txt", "w")
    
    # creat evaluation data -- split file2 into target / source, save with 0.2 chance                                                      
    with open(args.saveto + 'file2_s.txt', 'r') as val:
        for line in val:
            src, tgt = line.split(',')
            thresh = random.randint(0, 4)
            if thresh > 3:
                source_val.write(src)
                target_val.write(tgt)
    val.close()
    source_val.close()
    target_val.close()
    
if __name__ == '__main__':
    main()
