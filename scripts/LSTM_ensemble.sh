#!/bin/bash 
#SBATCH --job-name="lstm_ensemble" 
#SBATCH --account=normal 
#SBATCH --partition=cpu_mosaic_low 
#SBATCH --time=4:00:00 
#SBATCH --mem=10GB
#SBATCH --mail-user=bmkuwaha@uwaterloo.ca 
#SBATCH --mail-type=END,FAIL 
#SBATCH --output=%x.out
#SBATCH --error=%x.err

echo -n  "Starting." 
module load julia/1.7.3

echo "Current user: $USER" 

for j in {1..10}
do
julia ./discrete_model_LSTM_allvars.jl US-NY 120 2 4 5
done
