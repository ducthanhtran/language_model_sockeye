import pytest

import mxnet as mx

from . import lm_inference


SWITCHBOARD_PATH = '/work/smt2/tran/work/language_model_sockeye/models/switchboard/output.lr-00003'
MAX_OUTPUT_LEN = 100
BATCH_SIZE = 12
CONTEXT = mx.cpu()


def test_loading_models():
    model_hiddenstate_output = lm_inference.load_models(context=CONTEXT,
                                                        max_output_len=MAX_OUTPUT_LEN,
                                                        batch_size=BATCH_SIZE,
                                                        model_folder=SWITCHBOARD_PATH,
                                                        decoder_return_logit_inputs=True)

    model_softmax_output = lm_inference.load_models(context=CONTEXT,
                                                    max_output_len=MAX_OUTPUT_LEN,
                                                    batch_size=BATCH_SIZE,
                                                    model_folder=SWITCHBOARD_PATH,
                                                    decoder_return_logit_inputs=False)

def test_inference_softmax():
    model_softmax_output = lm_inference.load_models(context=CONTEXT,
                                                    max_output_len=MAX_OUTPUT_LEN,
                                                    batch_size=BATCH_SIZE,
                                                    model_folder=SWITCHBOARD_PATH,
                                                    decoder_return_logit_inputs=False)
    model.
