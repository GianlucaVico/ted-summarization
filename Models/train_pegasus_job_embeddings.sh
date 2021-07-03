#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=12:00:00
#SBATCH -A um_dke
#SBATCH --job-name="train_pegasus"
#SBATCH --output="pegasus_log/pegasus_embeddings_%A.log"
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
python3 test_gpu.py

python3 train_pegasus.py \
    -e 200 \
    -b 2 \
    -tb 2 \
    -l $WORK/pegasus_logs \
    -o $HPCWORK/pegasus_freeze_no_embeddings \
    --train-x $DATA/filter_train_documents.pkl \
    --train-y $DATA/filter_train_targets.pkl \
    --test-x $DATA/filter_test_documents.pkl \
    --test-y $DATA/filter_test_targets.pkl \
    -d cuda \
    --dropout 0.1 \
    -f ne \
    -ml 256 \
    --easy \
    -p 4096
#    -lp 2 \
#    -f
#    --do-sample \
#    --top-p 0.95 \
#    --top-k 100

#python3 train_pegasus.py \
#    -e 200 \
#    -b 4 \
#    -tb 4 \
#    -l $WORK/pegasus_logs \
#    -o $HPCWORK/pegasus \
#    --train-x $DATA/filter_train_documents.pkl \
#    --train-y $DATA/filter_train_targets.pkl \
#    --test-x $DATA/filter_test_documents.pkl \
#    --test-y $DATA/filter_test_targets.pkl \
#    -d cuda \
#    --dropout 0.1
#    --resume
