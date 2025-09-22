#!/bin/bash
# SLURM Template for Compute Jobs
# Replace placeholders with your actual values

#SBATCH --job-name=<JOB_NAME>            
#SBATCH --nodes=<NUMBER_OF_NODES>                     
#SBATCH --ntasks-per-node=<TASKS_PER_NODE>       
#SBATCH --gpus-per-task=<GPUS_PER_TASK>                     
#SBATCH --cpus-per-task=<CPUS_PER_TASK>          
#SBATCH --partition=<PARTITION_NAME>             
#SBATCH --time=<TIME_LIMIT>                        
#SBATCH --output=logs/<JOB_NAME>_%j.out      
#SBATCH --error=logs/<JOB_NAME>_%j.err       
#SBATCH --account=<ACCOUNT_NAME>

# Load required modules (modify as needed)
module load nccl/2.19.3-1
module load cuda/12.4.1

# Activate your virtual environment
source <PATH_TO_VENV>/bin/activate
which python 

# Set configuration file path
CONFIG_FILE="<PATH_TO_CONFIG_FILE>"

# Run the compute script

# For GPU processing (PyArrow-based):
python src/compute_arrow.py \
    --config "${CONFIG_FILE}"