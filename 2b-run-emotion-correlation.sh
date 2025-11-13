#!/bin/bash
#SBATCH --job-name=emotion-correlation
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
# Emotion Correlation Analysis: Extract CAP timeseries and correlate with emotion ratings
# Submit with: sbatch 2b-run-emotion-correlation.sh

pwd; hostname; date
set -e

# Create log directory if it doesn't exist
mkdir -p /home/data/nbc/misc-projects/diva-emotion-dfc/log/emotion-correlation
echo "Created/verified log directory: /home/data/nbc/misc-projects/diva-emotion-dfc/log/emotion-correlation"

echo "==================================================="
echo "STARTING EMOTION CORRELATION ANALYSIS"
echo "==================================================="

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

# Navigate to project directory
cd /home/data/nbc/misc-projects/diva-emotion-dfc

# Run emotion correlation analysis
cmd="python -u ./2b-emotion-correlation.py"

echo "Commandline: $cmd"
eval $cmd

echo "==================================================="
echo "EMOTION CORRELATION ANALYSIS COMPLETED"
echo "Results saved to: ./dset/derivatives/caps/emotion-correlation/"
echo "==================================================="

date