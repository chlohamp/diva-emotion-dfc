#!/bin/bash
#SBATCH --job-name=caps-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
#SBATCH --time=04:00:00
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/diva-emotion-dfc/log/%x/%x_%j.out
#SBATCH --error=/home/data/nbc/misc-projects/diva-emotion-dfc/log/%x/%x_%j.err
# ------------------------------------------
# CAP Analysis: Extract timeseries and perform k-means clustering
# Submit with: sbatch 2a-run-caps.sh

pwd; hostname; date
set -e

# Create log directory if it doesn't exist
mkdir -p /home/data/nbc/misc-projects/diva-emotion-dfc/log/caps-analysis
echo "Created/verified log directory: /home/data/nbc/misc-projects/diva-emotion-dfc/log/caps-analysis"

echo "==================================================="
echo "STARTING CAP ANALYSIS"
echo "==================================================="

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

# Navigate to project directory
cd /home/data/nbc/misc-projects/diva-emotion-dfc

# Run CAP analysis
cmd="python -u ./2a-caps.py"

echo "Commandline: $cmd"
eval $cmd

echo "==================================================="
echo "CAP ANALYSIS COMPLETED"
echo "Results saved to: ./dset/derivatives/caps/"
echo "==================================================="

date
