#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --mail-user=mac0c@glogin.ibex.kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=12:30:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --constraint=[gpu]&[local_200G]

#run the application:
srun run_fb15k.sh
