import os
import sys
from typing import Callable, List, Tuple, Union, Optional

import mxnet as mx
import numpy as np

from . import lm_common
from . import lm_model

sys.path.append('../')

import sockeye.constants as C
from sockeye import utils
from sockeye.vocab import Vocab, load_source_vocabs, vocab_from_json, are_identical



class ModelState:
    """
    A ModelState encapsulates information about the decoder states of an InferenceModel.
    """
    def __init__(self, states: List[mx.nd.NDArray]) -> None:
        self.states = states

    def sort_state(self, best_hyp_indices: mx.nd.NDArray):
        """
        Sorts states according to k-best order from last step in beam search.
        """
        self.states = [mx.nd.take(ds, best_hyp_indices) for ds in self.states]


class InferenceModel(lm_model.LanguageModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param config: Configuration object holding details about the model.
    :param params_fname: File with model parameters.
    :param context: MXNet context to bind modules to.
    :param batch_size: Batch size.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    """
    def __init__(self,
                 config: lm_common.LMConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 batch_size: int,
                 softmax_temperature: Optional[float] = None,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.batch_size = batch_size
        self.softmax_temperature = softmax_temperature

        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[int]
        self.decoder_data_shapes_cache = None  # type: Optional[Dict]
        self.decoder_return_logit_inputs = decoder_return_logit_inputs

        self.cache_output_layer_w_b = cache_output_layer_w_b
        self.output_layer_w = None  # type: Optional[mx.nd.NDArray]
        self.output_layer_b = None  # type: Optional[mx.nd.NDArray]

    def initialize(self, max_output_len: int):
        """
        Delayed construction of module.
        Originally to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_output_len: Maximum output length.
        """
        self.max_output_len = max_output_len
        self.decoder_module, self.decoder_default_bucket_key = self._get_decoder_module()

        self.decoder_data_shapes_cache = dict()  # bucket_key -> shape cache
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key)
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.params_fname)
        self.decoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)

        if self.cache_output_layer_w_b:
            if self.output_layer.weight_normalization:
                # precompute normalized output layer weight imperatively
                assert self.output_layer.weight_norm is not None
                weight = self.params[self.output_layer.weight_norm.weight.name].as_in_context(self.context)
                scale = self.params[self.output_layer.weight_norm.scale.name].as_in_context(self.context)
                self.output_layer_w = self.output_layer.weight_norm(weight, scale)
            else:
                self.output_layer_w = self.params[self.output_layer.w.name].as_in_context(self.context)
            self.output_layer_b = self.params[self.output_layer.b.name].as_in_context(self.context)

    def _get_decoder_module(self) -> Tuple[mx.mod.BucketingModule, int]:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the current length of the target sequences.

        :return: Tuple of decoder module and default bucket key.
        """

        def sym_gen(bucket_key: int):
            """
            Returns either softmax output (probs over target vocabulary) or inputs to logit
            computation, controlled by decoder_return_logit_inputs
            """
            decode_step = bucket_key

            self.decoder.reset()
            target_prev = mx.sym.Variable(C.TARGET_NAME)
            states = self.decoder.state_variables(decode_step)
            state_names = [state.name for state in states]

            # embedding for previous word
            # (batch_size, num_embed)
            target_embed_prev, _, _ = self.embedding.encode(data=target_prev, data_length=None, seq_len=1)

            # decoder
            # target_decoded: (batch_size, decoder_depth)
            (target_decoded,
             states) = self.decoder.decode_step(decode_step,
                                                target_embed_prev,
                                                *states)

            if self.decoder_return_logit_inputs:
                # skip output layer in graph
                outputs = mx.sym.identity(target_decoded, name=C.LOGIT_INPUTS_NAME)
            else:
                # logits: (batch_size, target_vocab_size)
                logits = self.output_layer(target_decoded)
                if self.softmax_temperature is not None:
                    logits /= self.softmax_temperature
                outputs = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            data_names = [C.TARGET_NAME] + state_names
            label_names = []  # type: List[str]
            return outputs, data_names, label_names

        # pylint: disable=not-callable
        default_bucket_key = self.max_output_len
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_decoder_data_shapes(self, bucket_key: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.
        Caches results for bucket_keys if called iteratively.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """
        target_max_length = bucket_key
        return self.decoder_data_shapes_cache.setdefault(
            bucket_key,
            [mx.io.DataDesc(name=C.TARGET_NAME, shape=(self.batch_size,), layout="NT")] +
            self.decoder.state_shapes(self.batch_size,
                                      target_max_length))

    def run_decoder(self,
                    prev_word: mx.nd.NDArray,
                    bucket_key: int,
                    model_state: 'ModelState') -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState']:
        """
        Runs forward pass of the single-step decoder.

        :return: Decoder stack output (logit inputs or probability distribution), attention scores, updated model state.
        """
        batch = mx.io.DataBatch(
            data=[prev_word.as_in_context(self.context)] + model_state.states,
            label=None,
            bucket_key=bucket_key,
            provide_data=self._get_decoder_data_shapes(bucket_key))
        self.decoder_module.forward(data_batch=batch, is_train=False)
        out, *model_state.states = self.decoder_module.get_outputs()
        return out, model_state

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        return self.config.max_seq_len_target

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.decoder.get_max_seq_len()


def load_model(context: mx.context.Context,
                max_output_len: Optional[int],
                batch_size: int,
                model_folder: List[str],
                checkpoint: Optional[int] = None,
                softmax_temperature: Optional[float] = None,
                decoder_return_logit_inputs: bool = False,
                cache_output_layer_w_b: bool = False) -> Tuple[InferenceModel,
                                                               Vocab]:
    """
    Loads a model for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param batch_size: Batch size.
    :param model_folders: List of model folders to load models from.
    :param checkpoint: Checkpoint to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param decoder_return_logit_inputs: Model decoders return inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Models cache weights and biases for logit computation as NumPy arrays (used with
                                   restrict lexicon).
    :return: Model, target vocabulary.
    """

    target_vocab = vocab_from_json(os.path.join(model_folder, lm_common.LM_PREFIX + lm_common.LM_VOCAB_NAME))
    model_config = lm_model.LanguageModel.load_config(os.path.join(model_folder, lm_common.LM_PREFIX + C.CONFIG_NAME))

    if checkpoint is None:
        params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
    else:
        params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

    inference_model = InferenceModel(config=model_config,
                                     params_fname=params_fname,
                                     context=context,
                                     batch_size=batch_size,
                                     softmax_temperature=softmax_temperature,
                                     decoder_return_logit_inputs=decoder_return_logit_inputs,
                                     cache_output_layer_w_b=cache_output_layer_w_b)

    inference_model.initialize(max_output_len)

    return inference_model, target_vocab


class LMInferer:
    """
    Final wrapper. Uses exactly two inference models, the first one
    using a softmax output and the second model gives
    us the hidden state of the RNN decoder.
    """
    def __init__(self,
                 context: mx.context.Context,
                 max_output_len: Optional[int], # really optional? Might need hard default value
                 batch_size: int,
                 model_folder: str,
                 checkpoint: Optional[int] = None,
                 softmax_temperature: Optional[float] = None,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False) -> None:
        """
        :param decoder_return_logit_inputs: If set to true we obtain hidden state outputs. Otherwise a softmax vector
                                            is returned by the inference model.
        """
        self.model, self.vocab = load_model(context, max_output_len, batch_size, model_folder, checkpoint, softmax_temperature, decoder_return_logit_inputs, cache_output_layer_w_b)

    def decode_step(self,
                    lm_states: ModelState,
                    sentence: mx.nd.NDArray,
                    step: int) -> Tuple[Union[mx.nd.NDArray, ModelState]]:
        """
        :param sentence: single array of integers denoting a sentence string
        :param step: current step of predicting a word. The previous word is located at position step-1 of
                     sentence.
        :return: output
        """
        prev_word = sentence[step-1]
        output, updated_lm_states = self.model.run_decoder(prev_word=prev_word,
                                                           bucket_key=step,
                                                           model_state=lm_states)
        return output, updated_lm_states
