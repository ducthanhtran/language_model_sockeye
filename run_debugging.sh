TRAIN="/work/smt2/tran/work/wmt18/tren/train-data.gz"
DEV="/work/smt2/tran/work/wmt18/tren/dev-data"

source /work/smt2/tran/work/virtualenvs/language_model/bin/activate

python -m language_model.lm_train --train-data "${TRAIN}" --dev-data "${DEV}"

deactivate
