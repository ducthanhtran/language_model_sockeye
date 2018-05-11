import pytest

import mxnet as mx

from . import lm_inference


SWITCHBOARD_PATH = '/work/smt2/tran/work/language_model_sockeye/models/switchboard/output.lr-00003'
VOC_SIZE = 1029

MAX_OUTPUT_LEN = 100
BATCH_SIZE = 1
CONTEXT = mx.cpu()

# according to vocabulary json file in SWITCHBOARD_PATH
SEQUENCE_BOS = mx.nd.array([2]) # '<s>'
SEQUENCE_ONE = mx.nd.array([2, 191]) # '<s> their'
SEQUENCE_TWO = mx.nd.array([2, 359, 595]) # '<s> same house'


def test_loading_models():
    model_hiddenstate_output = lm_inference.load_model(context=CONTEXT,
                                                        max_output_len=MAX_OUTPUT_LEN,
                                                        batch_size=BATCH_SIZE,
                                                        model_folder=SWITCHBOARD_PATH,
                                                        decoder_return_logit_inputs=True)

    # model_softmax_output = lm_inference.load_model(context=CONTEXT,
    #                                                 max_output_len=MAX_OUTPUT_LEN,
    #                                                 batch_size=BATCH_SIZE,
    #                                                 model_folder=SWITCHBOARD_PATH,
    #                                                 decoder_return_logit_inputs=False)


# def test_inference_softmax():
#     inferer = lm_inference.LMInferer(context=CONTEXT,
#                                      max_output_len=MAX_OUTPUT_LEN,
#                                      batch_size=BATCH_SIZE,
#                                      model_folder=SWITCHBOARD_PATH)
#
#     num_hidden = inferer.model.decoder.get_num_hidden()
#     zero_vector = mx.nd.zeros((1, num_hidden)) # same dimensionality as hidden units used during training, i.e. 1024
#     lm_state = lm_inference.ModelState([zero_vector, zero_vector, zero_vector, zero_vector])
#
#     def sentence_assertions_softmax(inferer, sentence, step):
#         out, updated_lm_states = inferer.decode_step(lm_states=lm_state,
#                                                         sentence=sentence,
#                                                         step=step)
#         assert out.shape[1] == VOC_SIZE
#         assert out.sum().asscalar() == pytest.approx(1.0, abs=1e-1) # out.sum() is an mx.nd.array - cast to int?
#
#         print("Sentence: {}".format(sentence))
#         print("softmax maximum value: {}".format(out.max()))
#         print("softmax argmax: {}".format(mx.ndarray.argmax(out, axis=1)))
#         print("updated lm_states: {}".format(updated_lm_states.states))# empty?? we did not update correctly
#
#     # sentence_assertions_softmax(inferer, SEQUENCE_BOS, step=1) # 'yeah' is argmax softmax output
#     # sentence_assertions_softmax(inferer, SEQUENCE_ONE, step=2) # ''
#     # sentence_assertions_softmax(inferer, SEQUENCE_TWO, step=3) # 'mall'
#
#
# def test_inference_hiddenstate():
#     inferer = lm_inference.LMInferer(context=CONTEXT,
#                                      max_output_len=MAX_OUTPUT_LEN,
#                                      batch_size=BATCH_SIZE,
#                                      model_folder=SWITCHBOARD_PATH,
#                                      decoder_return_logit_inputs=True)
#
#     num_hidden = inferer.model.decoder.get_num_hidden()
#     zero_vector = mx.nd.zeros((1, num_hidden)) # same dimensionality as hidden units used during training, i.e. 1024
#     lm_state = lm_inference.ModelState([zero_vector, zero_vector, zero_vector, zero_vector])
#
#     def sentence_assertions_hiddenstate(inferer, sentence, step):
#         out, updated_lm_states = inferer.decode_step(lm_states=lm_state,
#                                                         sentence=sentence,
#                                                         step=step)
#         assert out.shape[1] == inferer.model.decoder.get_num_hidden()
#
#         print("Sentence: {}".format(sentence))
#         print("output {}".format(out))
#         print("updated lm_states: {}".format(updated_lm_states.states)) # empty?? we did not update correctly
#
#     # sentence_assertions_hiddenstate(inferer, SEQUENCE_BOS, step=1) # 'yeah' is argmax softmax output
#     # sentence_assertions_hiddenstate(inferer, SEQUENCE_ONE, step=2) # ''
#     # sentence_assertions_hiddenstate(inferer, SEQUENCE_TWO, step=3) # 'mall'
