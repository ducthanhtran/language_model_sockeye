import pytest

import mxnet as mx

from . import lm_inference


SWITCHBOARD_PATH = '/work/smt2/tran/work/language_model_sockeye/models/switchboard/output.lr-00003'
VOC_SIZE = 1029

MAX_OUTPUT_LEN = 100
BATCH_SIZE = 1
CONTEXT = mx.cpu()

# according to vocabulary json file in SWITCHBOARD_PATH
SEQUENCE_BOS = mx.nd.array([2, 0, 0]) # '<s>'
SEQUENCE_ONE = mx.nd.array([2, 191, 0]) # '<s> their'
SEQUENCE_TWO = mx.nd.array([2, 359, 595]) # '<s> same house'

BATCH = mx.nd.stack(SEQUENCE_BOS, SEQUENCE_ONE, SEQUENCE_TWO, SEQUENCE_TWO)

def test_loading_models():
    model_hiddenstate_output = lm_inference.load_model(context=CONTEXT,
                                                        max_output_len=MAX_OUTPUT_LEN,
                                                        batch_size=BATCH_SIZE,
                                                        model_folder=SWITCHBOARD_PATH,
                                                        decoder_return_logit_inputs=True)

    model_softmax_output = lm_inference.load_model(context=CONTEXT,
                                                    max_output_len=MAX_OUTPUT_LEN,
                                                    batch_size=BATCH_SIZE,
                                                    model_folder=SWITCHBOARD_PATH,
                                                    decoder_return_logit_inputs=False)


def test_inference_softmax():
    inferer = lm_inference.LMInferer(context=CONTEXT,
                                     max_output_len=MAX_OUTPUT_LEN,
                                     batch_size=BATCH_SIZE,
                                     model_folder=SWITCHBOARD_PATH)
    # prepare LM states
    num_hidden = inferer.model.decoder.get_num_hidden()
    zero_vector = mx.nd.zeros((1, num_hidden)) # same dimensionality as hidden units used during training, i.e. 1024
    lm_state_zero = lm_inference.ModelState([zero_vector, zero_vector, zero_vector, zero_vector])
    one_vector = mx.nd.ones((1, num_hidden))
    lm_state_one = lm_inference.ModelState([one_vector, one_vector, one_vector, one_vector])
    prev_lm_states = [lm_state_zero, lm_state_zero, lm_state_zero, lm_state_one]

    steps = [1,2,3,3]

    out, updated_lm_states = inferer.decode_step(batch=BATCH,
                                                 prev_lm_states=prev_lm_states,
                                                 steps=steps)
    print("Output shape: {}".format(out.shape))
    print("Output softmax: {}".format(out))
    # assert out.shape == (BATCH.shape[0], 1, VOC_SIZE)
    # assert out.sum(axis=3).asscalar() == pytest.approx(1.0, abs=1e-1) # out.sum() is an mx.nd.array - cast to int?

    # print("softmax maximum value: {}".format(out.max()))
    # print("softmax argmax: {}".format(mx.ndarray.argmax(out, axis=1)))
    # print("updated lm_states: {}".format(updated_lm_states.states))# empty?? we did not update correctly
    # print("updated lm_states count: {}".format(len(updated_lm_states.states)))

# def test_inference_hiddenstate():
#     inferer = lm_inference.LMInferer(context=CONTEXT,
#                                      max_output_len=MAX_OUTPUT_LEN,
#                                      batch_size=BATCH_SIZE,
#                                      model_folder=SWITCHBOARD_PATH,
#                                      decoder_return_logit_inputs=True)
#     # prepare LM states
#     num_hidden = inferer.model.decoder.get_num_hidden()
#     zero_vector = mx.nd.zeros((1, num_hidden)) # same dimensionality as hidden units used during training, i.e. 1024
#     lm_state_zero = lm_inference.ModelState([zero_vector, zero_vector, zero_vector, zero_vector])
#     one_vector = mx.nd.ones((1, num_hidden))
#     lm_state_one = lm_inference.ModelState([one_vector, one_vector, one_vector, one_vector])
#     prev_lm_states = [lm_state_zero, lm_state_zero, lm_state_zero, lm_state_one]
#
#     steps = [1,2,3,3]
#
#     out, updated_lm_states = inferer.decode_step(batch=BATCH,
#                                                  prev_lm_states=prev_lm_states,
#                                                  steps=steps)
#     assert out.shape == (BATCH.shape[0], 1, inferer.model.decoder.get_num_hidden())
#
#     for lm_state in updated_lm_states:
#         assert lm_state.states # should not be empty
#
#     print("output {}".format(out))
