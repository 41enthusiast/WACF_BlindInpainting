#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu:pascal:1
 
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=long-high-prio # Use the short QOS
#SBATCH -t 7-0 # Set maximum walltime to 1 day
#SBATCH --job-name=omniwavnet_inpainting_test # Name of the job
#SBATCH --mem=16G # Request 16Gb of memory

#SBATCH -o program_output.txt
#SBATCH -e whoopsies.txt

# Load the global bash profile
source /etc/profile
module unload cuda
module load cuda/11.8

# Load your Python environment
source weath_diff/bin/activate

# Run the code
# python main.py --data_path_test ../art_painting/test/ --data_path ../art_painting/train/ --data_name art_painting
python3 train.py