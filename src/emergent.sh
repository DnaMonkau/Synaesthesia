#!/bin/bash
#SBATCH --job-name=izhikevich
#SBATCH --nodes=1
#SBATCH --time=0-10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --partition=batch
#SBATCH --mem=128GB
hostname
module load anaconda3
python emergent.py
