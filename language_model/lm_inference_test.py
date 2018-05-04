import pytest

import mxnet as mx

from . import lm_inference


SWITCHBOARD_PATH = '/work/smt2/tran/work/language_model_sockeye/models/switchboard/output.lr-00003'
MAX_INPUT_LEN = 75
BATCH_SIZE = 12
CONTEXT = mx.cpu()


def test_loading():
    model_hiddenstate_output = lm_inference.load_models(context=CONTEXT,
                                                        max_input_len=MAX_INPUT_LEN,
                                                        batch_size=BATCH_SIZE,
                                                        model_folders=[SWITCHBOARD_PATH],
                                                        decoder_return_logit_inputs=False)

    model_softmax_output = lm_inference.load_models(context=CONTEXT,
                                                    max_input_len=MAX_INPUT_LEN,
                                                    batch_size=BATCH_SIZE,
                                                    model_folders=[SWITCHBOARD_PATH],
                                                    decoder_return_logit_inputs=False)
