#!/bin/bash                                                                                                                                                                                                                                  
#                                                                                                                                                                                                                                         
#SBATCH --job-name=test                                                                                                                                                                                                                     
#SBATCH --nodes=1                                                                                                                                                                                                                           
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                    
#SBATCH --time=50:59:00                                                                                                                                                                                                                      
#SBATCH --mem=50GB                                                                                                                                                                                                                            
module purge
module load pytorch/python3.5/0.2.0_3

python3 -u process_data.py
