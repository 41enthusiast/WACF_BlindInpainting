#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu:ampere:1
 
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=long-high-prio # Use the short QOS
#SBATCH -t 7-0 # Set maximum walltime to 1 day
#SBATCH --job-name=wd_inpainting_test # Name of the job
#SBATCH --mem=16G # Request 16Gb of memory

#SBATCH -o program_output.txt
#SBATCH -e whoopsies.txt

# Load the global bash profile
source /etc/profile
module unload cuda
module load cuda/12.4

# Load your Python environment
source test1_agbi/bin/activate

# Run the code
python3 train.py