#!/bin/bash
#SBATCH --job-name=compute            
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1       
#SBATCH --gpus-per-task=1                     
#SBATCH --cpus-per-task=60          
#SBATCH --partition=nextgen             
#SBATCH --time=00:10:00                        
#SBATCH --output=logs/compute_%j.out      
#SBATCH --error=logs/compute_%j.err       
#SBATCH --account=PAS2136

module load nccl/2.19.3-1
module load cuda/12.4.1

source /fs/scratch/PAS2136/netzissou/venv/ascend_inference/bin/activate
which python 

CONFIG_FILE="/users/PZS1008/netzhang/emb-explorer/compute_config.json"

python src/compute_arrow.py \
    --config "${CONFIG_FILE}"
