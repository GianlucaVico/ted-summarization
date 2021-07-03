#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32000
#SBATCH --time=24:00:00
#SBATCH -A um_dke
#SBATCH --job-name="cascade"
#SBATCH --output="cascade_%A.log"
#SBATCH --gres=gpu:pascal:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_email@email.com

module switch intel gcc/9
module load python/3.8.7
module load cuda/110
module load cudnn/8.0.5
module load cmake
module load LIBRARIES
module load intelmkl

export PATH=$HOME/.local/bin:$PATH
export KENLM_ROOT=$HOME/kenlm

cd $HOME/Documents/BSc-Thesis-AudioSumm/Models

DATA=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset
MUSTC=$WORK/MUST-C/en-cs/data
TEDDATA=$WORK/TED/Data
TED=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED
AMARADATA=$WORK/AMARA
AMARA=$WORK/AMARA
DATASET=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/integrated_data.csv
#SUMM=$HPCWORK/pegasus_filter_hard_final/checkpoint-4460 # no punctuation no capitalisation
#SUMM=$HPCWORK/pegasus_real/checkpoint-3000-best # transcripts, change number
#ASR=$HPCWORK/wav2vec2_mustc/checkpoint-15000-best # w2v2 on mustc
ASR=$HPCWORK/wav2vec2/checkpoint-25000

nvidia-smi
python3 test_cascade.py \
    -d cuda \
    --ted $TED \
    --ted-data $TEDDATA \
    --amara $AMARA \
    --amara-data $AMARADATA \
    --must-c $MUSTC \
    --dataset $DATASET \
    --saved-audio $HPCWORK/cascade \
    --saved-summ $HPCWORK/cascade \
    --asr-model $ASR \
 #   --summ-model $SUMM \
 #  --asr-model $ASR \
