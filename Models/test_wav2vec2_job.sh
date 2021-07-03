#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000
#SBATCH --time=7:00:00
#SBATCH -A um_dke
#SBATCH --job-name="test_wav2vec2"
#SBATCH --output="wav2vec2_log/test_wav2vec2_mustc_kenlm_lf_4ng_%A.log"
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
export KENLM_ROOT=$HOME/kenlm

cd $HOME/Documents/BSc-Thesis-AudioSumm/Models

DATA=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset
MUSTC=$WORK/MUST-C/en-cs/data
TEDDATA=$WORK/TED/Data
TED=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED
AMARADATA=$WORK/AMARA
AMARA=$WORK/AMARA
DATASET=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/integrated_data.csv
MODEL=$HPCWORK/wav2vec2_mustc/checkpoint-15000-best

nvidia-smi
#echo 4-ngrams, no spell correction
echo 15-grams, no spell correction
python3 train_wav2vec2.py \
    -e 100000 \
    -b 16 \
    -tb 16 \
    -l $WORK/asr_logs \
    -o $HPCWORK/wav2vec2_mustc \
    -d cuda \
    -t $TED \
    --ted-data $TEDDATA \
    --amara $AMARA \
    --amara-data $AMARADATA \
    --must-c $MUSTC \
    --dataset $DATASET \
    --test-output $WORK/wav2vec2 \
    -f 0 \
    --saved-train $HPCWORK/mustc_train_audio \
    --saved-test $HPCWORK/mustc_test_audio \
    --length 128162 \
    --evaluate \
    --decoder viterbi \
    --lm $HPCWORK/kenlm_data/lm_train_chars_15ng_c.bin \
    --model $MODEL \
    --spell

#echo 4-ngrams, with spell correction
#python3 train_wav2vec2.py \
#    -e 100000 \
#    -b 16 \
#    -tb 32 \
#    -l $WORK/asr_logs \
#    -o $HPCWORK/wav2vec2_mustc \
#    -d cuda \
#    -t $TED \
#    --ted-data $TEDDATA \
#    --amara $AMARA \
#    --amara-data $AMARADATA \
#    --must-c $MUSTC \
#    --dataset $DATASET \
#    --test-output $WORK/wav2vec2 \
#    -f 0 \
#    --saved-train $HPCWORK/mustc_train_audio \
#    --saved-test $HPCWORK/mustc_test_audio \
#    --length 128162 \
#    -r \
#    --evaluate \
#    --decoder kenlm_lf \
#    --lm $HPCWORK/lm_wsj_kenlm_word_4g.bin \
#    --spell
#
#echo 15-ngrams (char), with spell correction
#python3 train_wav2vec2.py \
#    -e 100000 \
#    -b 16 \
#    -tb 32 \
#    -l $WORK/asr_logs \
#    -o $HPCWORK/wav2vec2_mustc \
#    -d cuda \
#    -t $TED \
#    --ted-data $TEDDATA \
#    --amara $AMARA \
#    --amara-data $AMARADATA \
#    --must-c $MUSTC \
#    --dataset $DATASET \
#    --test-output $WORK/wav2vec2 \
#    -f 0 \
#    --saved-train $HPCWORK/mustc_train_audio \
#    --saved-test $HPCWORK/mustc_test_audio \
#    --length 128162 \
#    -r \
#    --evaluate \
#    --decoder kenlm_lf \
#    --lm $HPCWORK/lm_wsj_kenlm_char_15g_pruned.bin \
#    --spell
