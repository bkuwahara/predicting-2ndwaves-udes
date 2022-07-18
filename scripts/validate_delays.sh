#!/bin/bash 
#SBATCH --job-name="delay_validation" 
#SBATCH --account=normal 
#SBATCH --partition=cpu_mosaic_low 
#SBATCH --time=24:00:00 
#SBATCH --mem=99GB
#SBATCH --mail-user=bmkuwaha@uwaterloo.ca 
#SBATCH --mail-type=END,FAIL 
#SBATCH --output=%x.out
#SBATCH --error=%x.err

echo -n  "Starting." 
module load julia/1.6.1

echo "Current user: $USER" 

for region in US-NY CA-ON UK
do
for taum in 10 14 21
do
for taur in 10 14
do
julia ./validate_model.jl $region $taum $taur
done
done
done
