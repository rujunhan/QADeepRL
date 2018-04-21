#!/bin/bash                                                                                                        
#                                                                                                                  
#SBATCH --job-name=test                                                                                            
#SBATCH --nodes=1                                                                                                  
#SBATCH --cpus-per-task=1                                                                                          
#SBATCH --time=50:59:00                                                                                                              
#SBATCH --mem=20GB                                                                                                                  
#SBATCH --gres=gpu:p40:1
  
module purge
module load pytorch/python3.6/0.3.0_4

python3 -u get_vocab.py
python3 -u preprocess.py -train_src /scratch/rjh347/input/source_trn_m.txt -train_tgt /scratch/rjh347/input/target_trn_m.txt -valid_src /scratch/rjh347/input/source_val_m.txt -valid_tgt /scratch/rjh347/input/target_val_m.txt -save_data /scratch/rjh347/monolingual/NMT0421 -max_shard_size 8388608 -src_vocab /scratch/rjh347/output/src_vcb.txt -tgt_vocab /scratch/rjh347/output/tgt_vcb.txt 

python3 -u train.py -data /scratch/rjh347/monolingual/NMT0421 -save_model /scratch/rjh347/monolingual/NMT0421 -train_from /scratch/rjh347/output/NMT0419_model_acc_59.07_ppl_7.59_e2.pt -gpuid 0
