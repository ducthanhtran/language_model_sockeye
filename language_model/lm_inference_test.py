import pytest

import mxnet as mx

from . import lm_inference


SWITCHBOARD_PATH = '/work/smt2/tran/work/language_model_sockeye/models/switchboard/output.lr-00003'
VOC_SIZE = 1029

MAX_OUTPUT_LEN = 100
BATCH_SIZE = 3
CONTEXT = mx.cpu()

# according to vocabulary json file in SWITCHBOARD_PATH
SEQUENCE_BOS = mx.nd.array([2, 11, 11]) # '<s> yeah yeah'
SEQUENCE_ONE = mx.nd.array([2, 80, 26]) # '<s> this is'
SEQUENCE_TWO = mx.nd.array([2, 4, 94]) # '<s> i go'

BATCH = mx.nd.stack(SEQUENCE_BOS, SEQUENCE_ONE, SEQUENCE_TWO)

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
    zero_vector = mx.nd.zeros((3, num_hidden)) # same dimensionality as hidden units used during training, i.e. 1024
    lm_state_zero = lm_inference.LMState([zero_vector, zero_vector, zero_vector, zero_vector])

    prev_lm_state = lm_state_zero
    step = 1

    out, updated_lm_state = inferer.decode_step(sequences=BATCH,
                                                state=prev_lm_state,
                                                step=step)

    print("Output shape: {}".format(out.shape))
    print("Output softmax: {}".format(out))

    assert out.shape == (BATCH.shape[0], VOC_SIZE)
#    assert out.sum(axis=1).asscalar() == pytest.approx(1.0, abs=1e-1) # out.sum() is an mx.nd.array - cast to int? # sum should be 1

    print("softmax sum: {}".format(out.sum(axis=1)))
    print("softmax maximum value: {}".format(out.max(axis=1)))
    print("softmax argmax: {}".format(mx.ndarray.argmax(out, axis=1)))
#    print("updated lm_states: {}".format(updated_lm_state.states))# empty?? we did not update correctly

    prev_lm_state = updated_lm_state
    step = 2

    out, updated_lm_state = inferer.decode_step(sequences=BATCH,
                                                state=prev_lm_state,
                                                step=step)

    print("Output shape: {}".format(out.shape))
    print("Output softmax: {}".format(out))
    print("softmax sum: {}".format(out.sum(axis=1)))
    print("softmax maximum value: {}".format(out.max(axis=1)))
    print("softmax argmax: {}".format(mx.ndarray.argmax(out, axis=1)))


def test_inference_hiddenstate():
    inferer = lm_inference.LMInferer(context=CONTEXT,
                                     max_output_len=MAX_OUTPUT_LEN,
                                     batch_size=BATCH_SIZE,
                                     model_folder=SWITCHBOARD_PATH,
                                     decoder_return_logit_inputs=True)
    # prepare LM states
    num_hidden = inferer.model.decoder.get_num_hidden()
    zero_vector = mx.nd.zeros((3, num_hidden)) # same dimensionality as hidden units used during training, i.e. 1024
    lm_state_zero = lm_inference.LMState([zero_vector, zero_vector, zero_vector, zero_vector])

    prev_lm_state = lm_state_zero
    step = 1

    out, updated_lm_state = inferer.decode_step(sequences=BATCH,
                                                state=prev_lm_state,
                                                step=step)
#    assert out.shape == (BATCH.shape[0], 1, inferer.model.decoder.get_num_hidden())

#    for lm_state in updated_lm_states:
#        assert lm_state.states # should not be empty

    print("output {}".format(out))
    print("update_state {}".format(updated_lm_state.states))

    step = 2
    out, updated_lm_state = inferer.decode_step(sequences=BATCH,
                                                state=updated_lm_state,
                                                step=step)

    print("output {}".format(out))
    print("update_state {}".format(updated_lm_state.states))
