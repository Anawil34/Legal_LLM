#!/bin/bash
#SBATCH -p gpu				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16          		# Specify number of nodes and processors per task
#SBATCH --gpus-per-task=4             # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH -t 4:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900304			# Specify project name
#SBATCH -J Trainmodel		# Specify job name

module reset
module load Mamba/23.11.0-0             # Load the module that you want to use
module load cudatoolkit/23.3_12.0
conda activate /path/.env/llamafactory

cd ./lib/LLaMA-Factory
llamafactory-cli train train_qwen2.yaml

