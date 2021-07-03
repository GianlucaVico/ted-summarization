#!/usr/local_rwth/bin/zsh

# Request a single task
#SBATCH --ntasks=1

#SBATCH --mem-per-cpu=16000

#SBATCH --time=8:00:00

# You must use the DKE project
#SBATCH -A um_dke

# Define the name of the job
#SBATCH --job-name="split_audio"

# Where to write output
#SBATCH --output="split_audio_fix.log"

#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_email@email.com

module switch intel gcc
module load python/3.8.7

export PATH=$HOME/.local/bin:$PATH

python3 split_audio.py
