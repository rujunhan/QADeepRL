#!/bin/bash                                                                                                        
#                                                                                                                  
#SBATCH --job-name=test                                                                                            
#SBATCH --nodes=1                                                                                                  
#SBATCH --cpus-per-task=1                                                                                          
#SBATCH --time=50:59:00                                                                                                              
#SBATCH --mem=100GB                                                                                                                  
#SBATCH --gres=gpu:p40:1
  
module purge
module load pytorch/python3.6/0.3.0_4

#python3 -u preprocess.py -train_src /scratch/rjh347/input/source_trn.txt -train_tgt /scratch/rjh347/input/target_trn.txt -valid_src /scratch/rjh347/input/source_val.txt -valid_tgt /scratch/rjh347/input/target_val.txt -save_data /scratch/rjh347/output/NMT0419 -max_shard_size 8388608

python3 -u train.py -data /scratch/rjh347/output/NMT0419 -save_model /scratch/rjh347/output/NMT0419_model -gpuid 0
