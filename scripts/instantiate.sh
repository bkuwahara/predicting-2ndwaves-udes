#!/bin/bash 
#SBATCH --job-name="instantiate" 
#SBATCH --account=normal 
#SBATCH --partition=cpu_mosaic_low 
#SBATCH --time=1:00:00 
#SBATCH --mem=10GB
#SBATCH --mail-user=bmkuwaha@uwaterloo.ca 
#SBATCH --mail-type=END,FAIL 
#SBATCH --output=%x.out
#SBATCH --error=%x.err

echo -n  "Starting." 
module --ignore_cache load julia/1.7.3

echo "Current user: $USER" 

julia ./instantiate.jl