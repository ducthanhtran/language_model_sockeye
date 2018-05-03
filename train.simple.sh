#!/bin/bash
###########################################################
# Paramters
#     1. output directory name
###########################################################
# Constants
MAIN_DIR="/work/smt2/tran/work/language_model_sockeye"
TRAIN="${MAIN_DIR}/data/ptb.test.txt"
DEV="${MAIN_DIR}/data/ptb.valid.txt"
OUTPUT="${MAIN_DIR}/models/${1}"
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
                                  --max-num-epochs 35 \
                                  --initial-learning-rate 1.0 \
                                  --learning-rate-reduce-factor 0.8 \
                                  --gradient-clipping-threshold 0.5 \
                                  --gradient-clipping-type norm \
                                  --num-embed 256 \
                                  --rnn-num-hidden 512 \
                                  --max-seq-len 80 \
                                  --rnn-forget-bias 1.0 \
                                  --optimizer sgd \
                                  --batch-size 256

deactivate
