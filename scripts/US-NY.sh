#!/bin/bash 
#SBATCH --job-name="US-NY" 
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

julia ./udde_model.jl US-NY 4 5
julia ./udde_model.jl US-NY 6