#!/bin/bash                                                                                                        
#                                                                                                                  
#SBATCH --job-name=test                                                                                            
#SBATCH --nodes=1                                                                                                  
#SBATCH --cpus-per-task=1                                                                                          
#SBATCH --time=50:59:00                                                                                                                                                                                             
#SBATCH --mem=132GB                                                                                                                                                                                                  

module purge
module load pytorch/python3.6/0.3.0_4

python3 -u process_data.py
python3 -u train_val_split.py
