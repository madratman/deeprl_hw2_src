#!/bin/bash

# Slurm Resource Parameters (Example)
#SBATCH -t 5-00:00              # Runtime in D-HH:MM
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --gres=gpu:1            # Number of gpus
#SBATCH -o q5_q6_%j.out      # File to which STDOUT will be written
#SBATCH -e q5_q6_%j.err      # File to which STDERR will be written
#SBATCH -w roberto                  # Partition to submit to

srun echo "I am on"
srun echo $HOSTNAME
srun echo "I got gpu number"
srun echo $CUDA_VISIBLE_DEVICES
srun echo "let the training begin"
srun nvidia-docker run --rm -e CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES` -v /data/datasets:/data/datasets -v /home/$USER:/home/$USER madratman/deeprl_hw /home/ratneshm/courses/deep_final/deeprl_hw2_src/slurm_final_2/q5_q6.sh