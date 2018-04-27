# Simple run script for debugging the language model training using WMT18 TR-EN data from newstest2017.
WMT18_DIR="/work/smt2/tran/work/wmt18/tren"
TRAIN="${WMT18_DIR}/train-newstest2017.100.gz" # 100 sentences
DEV="${WMT18_DIR}/dev-newstest2017"

OUTPUT="/work/smt2/tran/work/debug_output"


function file_exists {
  if [ ! -f $1 ]; then
      echo "File $1 not found!"
      exit 1
  fi
}


function directory_exists {
  if [ ! -f $1 ]; then
      echo "File $1 not found!"
      exit 1
  fi
}

###########################################################
## Main
file_exists "${TRAIN}"
file_exists "${DEV}"

source /work/smt2/tran/work/virtualenvs/language_model/bin/activate

python -m language_model.lm_train --train-data "${TRAIN}" --dev-data "${DEV}" --output "${OUTPUT}" --use-cpu

deactivate
