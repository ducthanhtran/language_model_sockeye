#!/bin/bash
###########################################################
# Paramters
#     1. output directory name
#     2. learning rate
###########################################################
# Constants
MAIN_DIR="/work/smt2/tran/work/language_model_sockeye"
TRAIN="${MAIN_DIR}/data/wmt2018/train.news.2016-2017.en.shuffled.pp.fc.bpe-tren-20k.concat.shuffled.maxlen-75.gz"
DEV="${MAIN_DIR}/data/wmt2018/dev.newsdev2016.tr-en.en.pp.fc.bpe.20k.joint"
OUTPUT="${MAIN_DIR}/models/wmt2018/${1}"
###########################################################

function file_exists {
  if [ ! -f "${1}" ]; then
      echo "File "${1}" not found!"
      exit 1
  fi
}


function create_if_directory_does_not_exists {
  if [ ! -d "${1}" ]; then
      mkdir -p "${1}"
  else
      echo "Output directory ${1} already exists. Exiting..."
      exit 1
  fi
}

###########################################################
## Main
file_exists "${TRAIN}"
file_exists "${DEV}"

create_if_directory_does_not_exists "${OUTPUT}"

source /work/smt2/tran/work/virtualenvs/language_model/bin/activate

python -m language_model.lm_train --train-data "${TRAIN}" \
                                  --dev-data "${DEV}" \
                                  --output "${OUTPUT}" \
                                  --max-num-epochs 80 \
                                  --initial-learning-rate "${2}" \
                                  --learning-rate-reduce-factor 0.8 \
                                  --gradient-clipping-threshold 2 \
                                  --gradient-clipping-type norm \
                                  --rnn-dropout-inputs 0.2 \
                                  --num-embed 512 \
                                  --rnn-num-hidden 2024 \
                                  --num-layers 2 \
                                  --batch-size 64 \
                                  --max-seq-len 75

deactivate
