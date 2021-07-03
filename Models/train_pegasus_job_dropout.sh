#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH -A um_dke
#SBATCH --job-name="pegasus_drop"
#SBATCH --output="pegasus_filter_dropout_2.log"
#SBATCH --gres=gpu:pascal:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_email@email.com

module switch intel gcc
module load python/3.8.7
module load cuda/110
module load cudnn/8.0.5

export PATH=$HOME/.local/bin:$PATH

cd $HOME/Documents/BSc-Thesis-AudioSumm/Models

DATA=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset

### Execute your application
nvidia-smi
#python3 test_gpu.py
python3 train_pegasus.py \
    -e 10 \
    -b 4 \
    -tb 4 \
    -l $WORK/pegasus_logs \
    -o $HPCWORK/pegasus_filter_dropout_2 \
    --train-x $DATA/filter_train_documents_no_string.pkl \
    --train-y $DATA/filter_train_targets_no_string.pkl \
    --test-x $DATA/filter_test_documents_no_string.pkl \
    --test-y $DATA/filter_test_targets_no_string.pkl \
    -d cuda \
    --dropout 0.2 \
    -ml 256 \
    --easy
    #--resume
