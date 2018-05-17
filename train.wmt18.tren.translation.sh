#!/bin/bash
###########################################################
# Constants
DATA_DIR="/work/smt2/tran/work/wmt18/tren/data"
TRAIN_S="${DATA_DIR}/train.en"
TRAIN_T="${DATA_DIR}/train.tr"
DEV_S="${DATA_DIR}/dev-newsdev2016.en"
DEV_T="${DATA_DIR}/dev-newsdev2016.tr"
OUTPUT="${DATA_DIR}/models/with_lm/output"
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
  fi
}

###########################################################
## Main
file_exists "${TRAIN_S}"
file_exists "${TRAIN_T}"
file_exists "${DEV_S}"
file_exists "${DEV_T}"

create_if_directory_does_not_exists "${OUTPUT}"

source /work/smt2/tran/work/virtualenvs/sockeye_latest_17th_may/bin/activate

python -m sockeye.train -s "${TRAIN_S}" \
                        -t "${TRAIN_T}" \
                        -vs "${DEV_S}" \
                        -vt "${DEV_T}" \
                        --batch-type word \
                        --batch-size 16384 \
                        --checkpoint-frequency 4000 \
                        --disable-device-locking \
                        --device-ids -1 \
                        --encoder transformer \
                        --decoder transformer \
                        --num-layers 6:6 \
                        --transformer-model-size 512 \
                        --transformer-attention-heads 8 \
                        --transformer-feed-forward-num-hidden 2048 \
                        --transformer-preprocess n \
                        --transformer-postprocess dr \
                        --transformer-dropout-prepost 0.3 \
                        --transformer-dropout-attention 0.3 \
                        --transformer-dropout-act 0.3 \
                        --embed-dropout 0.3:0.3 \
                        --max-seq-len 75:75 \
                        --label-smoothing 0.1 \
                        --weight-tying \
                        --weight-tying-type src_trg_softmax \
                        --num-embed 512:512 \
                        --initial-learning-rate 0.0001 \
                        --learning-rate-reduce-num-not-improved 3 \
                        --learning-rate-reduce-factor 0.7 \
                        --learning-rate-warmup 0 \
                        --max-num-checkpoint-not-improved 8 \
                        --weight-init-scale 3.0 \
                        --weight-init-xavier-factor-type avg \
                        --output "${OUTPUT}"

deactivate
