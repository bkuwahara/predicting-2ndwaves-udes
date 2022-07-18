#!/bin/bash 
#SBATCH --job-name="LSTM_halfwave" 
#SBATCH --account=normal 
#SBATCH --partition=cpu_mosaic_low 
#SBATCH --time=6:00:00 
#SBATCH --mem=20GB
#SBATCH --mail-user=bmkuwaha@uwaterloo.ca 
#SBATCH --mail-type=END,FAIL 
#SBATCH --output=%x.out
#SBATCH --error=%x.err

echo -n  "Starting." 
module load julia/1.7.3

echo "Current user: $USER" 


julia ./discrete_model_LSTM_allvars.jl CA-ON 56 4 5
julia ./discrete_model_LSTM_allvars.jl UK 48 4 5
julia ./discrete_model_LSTM_allvars.jl US-NY 44 4 5
