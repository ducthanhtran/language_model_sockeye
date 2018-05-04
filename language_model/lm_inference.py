import os
import sys
from typing import Callable, List, Tuple, Optional

import mxnet as mx
import numpy as np

from . import lm_common
from . import lm_model


sys.path.append('../')

import sockeye.constants as C
from sockeye.vocab import Vocab, load_source_vocabs, vocab_from_json


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
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.batch_size = batch_size
        self.softmax_temperature = softmax_temperature
        self.max_input_length, self.get_max_output_length = models_max_input_output_length([self],
                                                                                           max_output_length_num_stds)

        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[Tuple[int, int]]
        self.decoder_data_shapes_cache = None  # type: Optional[Dict]
        self.decoder_return_logit_inputs = decoder_return_logit_inputs

        self.cache_output_layer_w_b = cache_output_layer_w_b
        self.output_layer_w = None  # type: Optional[mx.nd.NDArray]
        self.output_layer_b = None  # type: Optional[mx.nd.NDArray]

    def initialize(self, max_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_input_length = max_input_length
        self.get_max_output_length = get_max_output_length_function

        # check the maximum supported length of the decoder:
        if self.max_supported_seq_len_target is not None:
            decoder_max_len = self.get_max_output_length(max_input_length)
            utils.check_condition(decoder_max_len <= self.max_supported_seq_len_target,
                                  "Decoder only supports a maximum length of %d, but %d was requested. Note that the "
                                  "maximum output length depends on the input length and the source/target length "
                                  "ratio observed during training." % (self.max_supported_seq_len_target,
                                                                       decoder_max_len))

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

    def _get_decoder_module(self) -> Tuple[mx.mod.BucketingModule, Tuple[int, int]]:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the length of the source sequence
        and the current time-step in the inference procedure (e.g. beam search).
        The latter corresponds to the current length of the target sequences.

        :return: Tuple of decoder module and default bucket key.
        """

        def sym_gen(bucket_key: Tuple[int, int]):
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
            return mx.sym.Group([outputs] + states), data_names, label_names

        # pylint: disable=not-callable
        default_bucket_key = (self.max_input_length, self.get_max_output_length(self.max_input_length))
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.
        Caches results for bucket_keys if called iteratively.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """
        source_max_length, target_max_length = bucket_key
        # TODO: state_shapes method has to be implemented in lm_decoder.py
        return self.decoder_data_shapes_cache.setdefault(
            bucket_key,
            [mx.io.DataDesc(name=C.TARGET_NAME, shape=(self.batch_size * self.beam_size,), layout="NT")] +
            self.decoder.state_shapes(self.batch_size * self.beam_size,
                                      target_max_length,
                                      self.encoder.get_encoded_seq_len(source_max_length),
                                      self.encoder.get_num_hidden()))

    def run_decoder(self,
                    prev_word: mx.nd.NDArray,
                    bucket_key: Tuple[int, int],
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


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                batch_size: int,
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None,
                max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                decoder_return_logit_inputs: bool = False,
                cache_output_layer_w_b: bool = False) -> Tuple[List[InferenceModel],
                                                               Vocab]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param batch_size: Batch size.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations to add to mean target-source length ratio
           to compute maximum output length.
    :param decoder_return_logit_inputs: Model decoders return inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Models cache weights and biases for logit computation as NumPy arrays (used with
                                   restrict lexicon).
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    """
    models = []  # type: List[InferenceModel]

    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        target_vocabs.append(vocab_from_json(os.path.join(model_folder, lm_common.LM_PREFIX + lm_common.LM_VOCAB_NAME)))
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
        models.append(inference_model)

    utils.check_condition(vocab.are_identical(*target_vocabs), "Target vocabulary ids do not match")

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = models_max_input_output_length(models,
                                                                          max_output_length_num_stds,
                                                                          max_input_len)
    for inference_model in models:
        inference_model.initialize(max_input_len, get_max_output_length)

    return models, target_vocabs[0]


def models_max_input_output_length(models: List[InferenceModel],
                                   num_stds: int,
                                   forced_max_input_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param forced_max_input_len: An optional overwrite of the maximum input length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    supported_max_seq_len_target = min((model.max_supported_seq_len_target for model in models
                                        if model.max_supported_seq_len_target is not None),
                                       default=None)

    training_max_seq_len_target = min(model.training_max_seq_len_target for model in models)

    return get_max_input_output_length(supported_max_seq_len_target,
                                       training_max_seq_len_target,
                                       forced_max_input_len=forced_max_input_len)


def get_max_input_output_length(supported_max_seq_len_target: Optional[int],
                                training_max_seq_len_target: Optional[int],
                                forced_max_input_len: Optional[int]) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length. It takes into account optional maximum source and target lengths.

    :param supported_max_seq_len_target: The maximum target length supported by the models.
    :param forced_max_input_len: An optional overwrite of the maximum input length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    space_for_bos = 1
    space_for_eos = 1

    factor = C.TARGET_MAX_LENGTH_FACTOR

    if forced_max_input_len is None:
        # Make sure that if there is a hard constraint on the maximum source or target length we never exceed this
        # constraint. This is for example the case for learned positional embeddings, which are only defined for the
        # maximum source and target sequence length observed during training.
        if supported_max_seq_len_target is not None:
            max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
            if np.ceil(factor * training_max_seq_len_target) > max_output_len:
                max_input_len = int(np.floor(max_output_len / factor))
            else:
                max_input_len = training_max_seq_len_target
        else:
            # we use the maximum length from training.
            max_input_len = training_max_seq_len_target
    else:
        max_input_len = forced_max_input_len

    def get_max_output_length(input_length: int):
        """
        Returns the maximum output length for inference given the input length.
        Explicitly includes space for BOS and EOS sentence symbols in the target sequence, because we assume
        that the mean length ratio computed on the training data do not include these special symbols.
        (see data_io.analyze_sequence_lengths)
        """

        return int(np.ceil(factor * input_length)) + space_for_bos + space_for_eos

    return max_input_len, get_max_output_length


class LMInferer:
    """
    Final wrapper. Uses exactly two inference models, the first one
    using a softmax output and the second model gives
    us the hidden state of the RNN decoder.
    """
    def __init__(self,
                 context: mx.context.Context,
                 target_vocab) -> None:
        """
        :param context: context for running computation.
        :param models: exactly two inference models, the first one gives us the softmax output and the second model
                       returns the hidden state from the RNN decoder.
        :param target_vocab: target vocabulary
        """
        self.context = context
        self.target_vocab = target_vocab


    def decode_step(self,
                    sequences: mx.nd.NDArray,
                    step: int,
                    states: Tuple[ModelState, ModelState]) -> Tuple[mx.nd.NDArray, List[ModelState]]:
        """
        Computes softmax and hidden state output given the previous word and previous hidden states. The new hidden state is simply appended to the states-list.

        :param sequences: sequences of current output hypotheses. Shape: (batch_size, max_output_length).
        :param step: current timestep
        :param states: exactly two ModelStates for each inference model
        """
        bucket_key = (source_length, step) # TODO: target length?
        prev_word = sequences[:, step-1]

        outputs = []
        model_states = []

        for model, state in zip(self.models, states):
            decoder_output, model_state_hidden = model.run_decoder(prev_word, bucket_key, state)
            outputs.append(decoder_output)
            model_states.append(model_state_hidden)

        # first model responsible for softmax and second for hidden state output
        return outputs[0], model_states[1][0]
