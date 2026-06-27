#!/bin/sh
#SBATCH -A uppmax2025-3-5
#SBATCH -p node
#SBATCH -M snowy
#SBATCH --gres=gpu:1   
#SBATCH --time=03:0:00 
#SBATCH -J lab2_seq2seq 
#SBATCH --mail-type=END
#SBATCH --mail-user=meriem.beloucif@lingfil.uu.se

export CUDA_VISIBLE_DEVICES=0 


# Activate your conda environment
source /home/beloucif/miniconda3/etc/profile.d/conda.sh   # adjust if conda is installed elsewhere, use find ~/ -name "conda.sh"  
conda activate mt25_b                        # replace with your env name

# Path to your Python file
PYTHON_SCRIPT="seq2seq.py"

# Run your Python script with arguments if needed
python $PYTHON_SCRIPT 



