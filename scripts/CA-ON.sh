#!/bin/bash 
#SBATCH --job-name="CA-ON" 
#SBATCH --account=normal 
#SBATCH --partition=cpu_mosaic_low 
#SBATCH --time=12:00:00 
#SBATCH --mem=99GB
#SBATCH --mail-user=bmkuwaha@uwaterloo.ca 
#SBATCH --mail-type=END,FAIL 
#SBATCH --output=%x.out
#SBATCH --error=%x.err

echo -n  "Starting." 
module load julia/1.6.1

echo "Current user: $USER" 

julia ./train_ude_delay.jl CA-ON
julia ./analysis.jl CA-ON 
