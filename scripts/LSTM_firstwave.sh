#!/bin/bash 
#SBATCH --job-name="LSTM_1stwave" 
#SBATCH --account=normal 
#SBATCH --partition=cpu_mosaic_low 
#SBATCH --time=1:00:00 
#SBATCH --mem=10GB
#SBATCH --mail-user=bmkuwaha@uwaterloo.ca 
#SBATCH --mail-type=END,FAIL 
#SBATCH --output=%x.out
#SBATCH --error=%x.err

echo -n  "Starting." 
module load julia/1.7.3

echo "Current user: $USER" 

for region in US-NY CA-ON UK
do
julia ./discrete_model_LSTM_allvars.jl $region 120 2 4 5
done

