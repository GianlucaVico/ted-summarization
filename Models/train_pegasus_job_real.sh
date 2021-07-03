#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12000
#SBATCH --time=24:00:00
#SBATCH -A um_dke
#SBATCH --job-name="peg_real"
#SBATCH --output="pegasus_log/pegasus_real_%A.log"
#SBATCH --gres=gpu:pascal:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_email@email.com

module switch intel gcc/9
module load python/3.8.7
module load cuda/110
module load cudnn/8.0.5
module load cmake
module load LIBRARIES
module load intelmkl/2020

export PATH=$HOME/.local/bin:$PATH

cd $HOME/Documents/BSc-Thesis-AudioSumm/Models

DATA=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset

### Execute your application
nvidia-smi
#python3 test_gpu.py
python3 train_pegasus.py \
    -e 30 \
    -b 4 \
    -tb 4 \
    -l $WORK/pegasus_logs \
    -o $HPCWORK/pegasus_real_2 \
    --train-x $DATA/train_transcript_documents.pkl \
    --train-y $DATA/train_transcript_targets.pkl \
    --test-x $DATA/test_transcript_documents.pkl \
    --test-y $DATA/test_transcript_targets.pkl \
    -d cuda \
    --dropout 0.1 \
    -ml 256 \
#    --model $HPCWORK/pegasus_real_2/checkpoint-2700
#    --model $HPCWORK/pegasus/checkpoint-2676
#    --model $HPCWORK/pegasus_filter_hard_final/checkpoint-4460
#    --easy \
#    -p 4096 \
#    --debug
