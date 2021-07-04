#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=500
#SBATCH --time=04:00:00
#SBATCH -A um_dke
#SBATCH --job-name="ted_download"
#SBATCH --output="ted_download_%A.log"
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=ganlucavico@gmail.co

module switch intel gcc
module load python/3.8.7

export PATH=$HOME/.local/bin:$PATH

cd $HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED

TED=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED
TOTAL=1
### Execute your application
python3 ted_audio.py -id $TED/talk_id.csv -d $TED/Data/data_urls.json -f $WORK/TED/Data
