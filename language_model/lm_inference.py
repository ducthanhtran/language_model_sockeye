import os
import sys
from typing import List, Tuple, Optional

import mxnet as mx

from . import lm_model

sys.path.append('../')

import sockeye.constants as C
from sockeye.model import SockeyeModel
from sockeye.vocab import Vocab, load_source_vocabs, vocab_from_json


BeamHistory = Dict[str, List]
Tokens = List[str]


class InferenceModel(SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param config: Configuration object holding details about the model.
    :param params_fname: File with model parameters.
    :param context: MXNet context to bind modules to.
    :param beam_size: Beam size.
    :param batch_size: Batch size.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 beam_size: int,
                 batch_size: int,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.beam_size = beam_size
        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')
        self.batch_size = batch_size
        self.softmax_temperature = softmax_temperature
        self.max_input_length, self.get_max_output_length = models_max_input_output_length([self],
                                                                                           max_output_length_num_stds)

        self.encoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.encoder_default_bucket_key = None  # type: Optional[int]
        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[Tuple[int, int]]
        self.decoder_data_shapes_cache = None  # type: Optional[Dict]
        self.decoder_return_logit_inputs = decoder_return_logit_inputs

        self.cache_output_layer_w_b = cache_output_layer_w_b
        self.output_layer_w = None  # type: Optional[mx.nd.NDArray]
        self.output_layer_b = None  # type: Optional[mx.nd.NDArray]

    @property
    def num_source_factors(self) -> int:
        """
        Returns the number of source factors of this InferenceModel (at least 1).
        """
        return self.config.config_data.num_source_factors

    def initialize(self, max_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_input_length = max_input_length
        if self.max_input_length > self.training_max_seq_len_source:
            logger.warning("Model was only trained with sentences up to a length of %d, "
                           "but a max_input_len of %d is used.",
                           self.training_max_seq_len_source, self.max_input_length)
        self.get_max_output_length = get_max_output_length_function

        # check the maximum supported length of the encoder & decoder:
        if self.max_supported_seq_len_source is not None:
            utils.check_condition(self.max_input_length <= self.max_supported_seq_len_source,
                                  "Encoder only supports a maximum length of %d" % self.max_supported_seq_len_source)
        if self.max_supported_seq_len_target is not None:
            decoder_max_len = self.get_max_output_length(max_input_length)
            utils.check_condition(decoder_max_len <= self.max_supported_seq_len_target,
                                  "Decoder only supports a maximum length of %d, but %d was requested. Note that the "
                                  "maximum output length depends on the input length and the source/target length "
                                  "ratio observed during training." % (self.max_supported_seq_len_target,
                                                                       decoder_max_len))

        self.encoder_module, self.encoder_default_bucket_key = self._get_encoder_module()
        self.decoder_module, self.decoder_default_bucket_key = self._get_decoder_module()

        self.decoder_data_shapes_cache = dict()  # bucket_key -> shape cache
        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.params_fname)
        self.encoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)
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

    def _get_encoder_module(self) -> Tuple[mx.mod.BucketingModule, int]:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Tuple of encoder module and default bucket key.
        """

        def sym_gen(source_seq_len: int):
            source = mx.sym.Variable(C.SOURCE_NAME)
            source_words = source.split(num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            source_length = utils.compute_lengths(source_words)

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # encoder
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # initial decoder states
            decoder_init_states = self.decoder.init_states(source_encoded,
                                                           source_encoded_length,
                                                           source_encoded_seq_len)

            data_names = [C.SOURCE_NAME]
            label_names = []  # type: List[str]
            return mx.sym.Group(decoder_init_states), data_names, label_names

        default_bucket_key = self.max_input_length
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

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
            source_seq_len, decode_step = bucket_key
            source_embed_seq_len = self.embedding_source.get_encoded_seq_len(source_seq_len)
            source_encoded_seq_len = self.encoder.get_encoded_seq_len(source_embed_seq_len)

            self.decoder.reset()
            target_prev = mx.sym.Variable(C.TARGET_NAME)
            states = self.decoder.state_variables(decode_step)
            state_names = [state.name for state in states]

            # embedding for previous word
            # (batch_size, num_embed)
            target_embed_prev, _, _ = self.embedding_target.encode(data=target_prev, data_length=None, seq_len=1)

            # decoder
            # target_decoded: (batch_size, decoder_depth)
            (target_decoded,
             attention_probs,
             states) = self.decoder.decode_step(decode_step,
                                                target_embed_prev,
                                                source_encoded_seq_len,
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
            return mx.sym.Group([outputs, attention_probs] + states), data_names, label_names

        # pylint: disable=not-callable
        default_bucket_key = (self.max_input_length, self.get_max_output_length(self.max_input_length))
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_encoder_data_shapes(self, bucket_key: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(self.batch_size, bucket_key, self.num_source_factors),
                               layout=C.BATCH_MAJOR)]

    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.
        Caches results for bucket_keys if called iteratively.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """
        source_max_length, target_max_length = bucket_key
        return self.decoder_data_shapes_cache.setdefault(
            bucket_key,
            [mx.io.DataDesc(name=C.TARGET_NAME, shape=(self.batch_size * self.beam_size,), layout="NT")] +
            self.decoder.state_shapes(self.batch_size * self.beam_size,
                                      target_max_length,
                                      self.encoder.get_encoded_seq_len(source_max_length),
                                      self.encoder.get_num_hidden()))

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int) -> 'ModelState':
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens. Shape (batch_size, source length, num_source_factors).
        :param source_max_length: Bucket key.
        :return: Initial model state.
        """
        batch = mx.io.DataBatch(data=[source],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_states = self.encoder_module.get_outputs()
        # replicate encoder/init module results beam size times
        decoder_states = [mx.nd.repeat(s, repeats=self.beam_size, axis=0) for s in decoder_states]
        return ModelState(decoder_states)

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
        out, attention_probs, *model_state.states = self.decoder_module.get_outputs()
        return out, attention_probs, model_state

    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.encoder.get_max_seq_len()

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.decoder.get_max_seq_len()

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                beam_size: int,
                batch_size: int,
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None,
                max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                decoder_return_logit_inputs: bool = False,
                cache_output_layer_w_b: bool = False) -> Tuple[List[InferenceModel],
                                                               List[Vocab],
                                                               Vocab]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
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
    source_vocabs = []  # type: List[List[vocab.Vocab]]
    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model_source_vocabs = load_source_vocabs(model_folder)
        source_vocabs.append(model_source_vocabs)
        target_vocabs.append(vocab_from_json(os.path.join(model_folder, C.VOCAB_TRG_NAME)))
        model_config = lm_model.load_config(os.path.join(model_folder, C.CONFIG_NAME))

        if checkpoint is None:
            params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

        inference_model = InferenceModel(config=model_config,
                                         params_fname=params_fname,
                                         context=context,
                                         beam_size=beam_size,
                                         batch_size=batch_size,
                                         softmax_temperature=softmax_temperature,
                                         decoder_return_logit_inputs=decoder_return_logit_inputs,
                                         cache_output_layer_w_b=cache_output_layer_w_b)
        utils.check_condition(inference_model.num_source_factors == len(model_source_vocabs),
                              "Number of loaded source vocabularies (%d) does not match "
                              "number of source factors for model '%s' (%d)" % (len(model_source_vocabs), model_folder,
                                                                                inference_model.num_source_factors))
        models.append(inference_model)

    utils.check_condition(vocab.are_identical(*target_vocabs), "Target vocabulary ids do not match")
    first_model_vocabs = source_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[source_vocabs[i][fi] for i in range(len(source_vocabs))]),
                              "Source vocabulary ids do not match. Factor %d" % fi)

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = models_max_input_output_length(models,
                                                                          max_output_length_num_stds,
                                                                          max_input_len)
    for inference_model in models:
        inference_model.initialize(max_input_len, get_max_output_length)

    return models, source_vocabs[0], target_vocabs[0]


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
    max_mean = max(model.length_ratio_mean for model in models)
    max_std = max(model.length_ratio_std for model in models)

    supported_max_seq_len_source = min((model.max_supported_seq_len_source for model in models
                                        if model.max_supported_seq_len_source is not None),
                                       default=None)
    supported_max_seq_len_target = min((model.max_supported_seq_len_target for model in models
                                        if model.max_supported_seq_len_target is not None),
                                       default=None)
    training_max_seq_len_source = min(model.training_max_seq_len_source for model in models)

    return get_max_input_output_length(supported_max_seq_len_source,
                                       supported_max_seq_len_target,
                                       training_max_seq_len_source,
                                       forced_max_input_len=forced_max_input_len,
                                       length_ratio_mean=max_mean,
                                       length_ratio_std=max_std,
                                       num_stds=num_stds)


    def get_max_input_output_length(supported_max_seq_len_source: Optional[int],
                                    supported_max_seq_len_target: Optional[int],
                                    training_max_seq_len_source: Optional[int],
                                    forced_max_input_len: Optional[int],
                                    length_ratio_mean: float,
                                    length_ratio_std: float,
                                    num_stds: int) -> Tuple[int, Callable]:
        """
        Returns a function to compute maximum output length given a fixed number of standard deviations as a
        safety margin, and the current input length. It takes into account optional maximum source and target lengths.

        :param supported_max_seq_len_source: The maximum source length supported by the models.
        :param supported_max_seq_len_target: The maximum target length supported by the models.
        :param training_max_seq_len_source: The maximum source length observed during training.
        :param forced_max_input_len: An optional overwrite of the maximum input length.
        :param length_ratio_mean: The mean of the length ratio that was calculated on the raw sequences with special
               symbols such as EOS or BOS.
        :param length_ratio_std: The standard deviation of the length ratio.
        :param num_stds: The number of standard deviations the target length may exceed the mean target length (as long as
               the supported maximum length allows for this).
        :return: The maximum input length and a function to get the output length given the input length.
        """
        space_for_bos = 1
        space_for_eos = 1

        if num_stds < 0:
            factor = C.TARGET_MAX_LENGTH_FACTOR  # type: float
        else:
            factor = length_ratio_mean + (length_ratio_std * num_stds)

        if forced_max_input_len is None:
            # Make sure that if there is a hard constraint on the maximum source or target length we never exceed this
            # constraint. This is for example the case for learned positional embeddings, which are only defined for the
            # maximum source and target sequence length observed during training.
            if supported_max_seq_len_source is not None and supported_max_seq_len_target is None:
                max_input_len = supported_max_seq_len_source
            elif supported_max_seq_len_source is None and supported_max_seq_len_target is not None:
                max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
                if np.ceil(factor * training_max_seq_len_source) > max_output_len:
                    max_input_len = int(np.floor(max_output_len / factor))
                else:
                    max_input_len = training_max_seq_len_source
            elif supported_max_seq_len_source is not None or supported_max_seq_len_target is not None:
                max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
                if np.ceil(factor * supported_max_seq_len_source) > max_output_len:
                    max_input_len = int(np.floor(max_output_len / factor))
                else:
                    max_input_len = supported_max_seq_len_source
            else:
                # Any source/target length is supported and max_input_len was not manually set, therefore we use the
                # maximum length from training.
                max_input_len = training_max_seq_len_source
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

class Translator:
    """
    Translator uses one or several models to translate input.
    It holds references to vocabularies to takes care of encoding input strings as word ids and conversion
    of target ids into a translation string.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param models: List of models.
    :param source_vocabs: Source vocabularies.
    :param target_vocab: Target vocabulary.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary restriction.
    :param store_beam: If True, store the beam search history and return it in the TranslatorOutput.
    """

    def __init__(self,
                 context: mx.context.Context,
                 ensemble_mode: str,
                 bucket_source_width: int,
                 length_penalty: LengthPenalty,
                 models: List[InferenceModel],
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 restrict_lexicon: Optional[lexicon.TopKLexicon] = None,
                 store_beam : bool = False) -> None:
        self.context = context
        self.length_penalty = length_penalty
        self.source_vocabs = source_vocabs
        self.vocab_target = target_vocab
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.restrict_lexicon = restrict_lexicon
        self.store_beam = store_beam
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        assert C.PAD_ID == 0, "pad id should be 0"
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}  # type: Set[int]
        self.models = models
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        self.batch_size = self.models[0].batch_size
        # after models are loaded we ensured that they agree on max_input_length, max_output_length and batch size
        self.max_input_length = self.models[0].max_input_length
        max_output_length = self.models[0].get_max_output_length(self.max_input_length)
        if bucket_source_width > 0:
            self.buckets_source = data_io.define_buckets(self.max_input_length, step=bucket_source_width)
        else:
            self.buckets_source = [self.max_input_length]
        self.pad_dist = mx.nd.full((self.batch_size * self.beam_size, len(self.vocab_target)), val=np.inf,
                                   ctx=self.context)
        logger.info("Translator (%d model(s) beam_size=%d ensemble_mode=%s batch_size=%d "
                    "buckets_source=%s)",
                    len(self.models),
                    self.beam_size,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.batch_size,
                    self.buckets_source)

    @property
    def num_source_factors(self) -> int:
        return self.models[0].num_source_factors

    @staticmethod
    def _get_interpolation_func(ensemble_mode):
        if ensemble_mode == 'linear':
            return Translator._linear_interpolation
        elif ensemble_mode == 'log_linear':
            return Translator._log_linear_interpolation
        else:
            raise ValueError("unknown interpolation type")

    @staticmethod
    def _linear_interpolation(predictions):
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = utils.average_arrays([mx.nd.log(p) for p in predictions])
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(mx.nd.softmax(log_probs))

    def translate(self, trans_inputs: List[TranslatorInput]) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Splits oversized sentences to sentence chunks of size less than max_input_length.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :return: List of translation results.
        """
        translated_chunks = []  # type: List[TranslatedChunk]

        # split into chunks
        input_chunks = []  # type: List[TranslatorInput]
        for input_idx, trans_input in enumerate(trans_inputs, 1):

            # bad input
            if isinstance(trans_input, BadTranslatorInput):
                translated_chunks.append(TranslatedChunk(id=input_idx, chunk_id=0, translation=empty_translation()))

            # empty input
            elif len(trans_input.tokens) == 0:
                translated_chunks.append(TranslatedChunk(id=input_idx, chunk_id=0, translation=empty_translation()))

            # oversized input
            elif len(trans_input.tokens) > self.max_input_length:
                logger.debug(
                    "Input %d has length (%d) that exceeds max input length (%d). Splitting into chunks of size %d.",
                    trans_input.sentence_id, len(trans_input.tokens), self.buckets_source[-1], self.max_input_length)
                input_chunks.extend(list(trans_input.chunks(self.max_input_length)))

            # regular input
            else:
                input_chunks.append(trans_input)

        # Sort longest to shortest (to rather fill batches of shorter than longer sequences)
        input_chunks = sorted(input_chunks, key=lambda chunk: len(chunk.tokens), reverse=True)

        # translate in batch-sized blocks over input chunks
        for batch_id, batch in enumerate(utils.grouper(input_chunks, self.batch_size)):
            logger.debug("Translating batch %d", batch_id)
            # underfilled batch will be filled to a full batch size with copies of the 1st input
            rest = self.batch_size - len(batch)
            if rest > 0:
                logger.debug("Extending the last batch to the full batch size (%d)", self.batch_size)
                batch = batch + [batch[0]] * rest
            batch_translations = self._translate_nd(*self._get_inference_input(batch))
            # truncate to remove filler translations
            if rest > 0:
                batch_translations = batch_translations[:-rest]
            for chunk, translation in zip(batch, batch_translations):
                translated_chunks.append(TranslatedChunk(chunk.sentence_id, chunk.chunk_id, translation))
        # Sort by input idx and then chunk id
        translated_chunks = sorted(translated_chunks)

        # Concatenate results
        results = []  # type: List[TranslatorOutput]
        chunks_by_input_idx = itertools.groupby(translated_chunks, key=lambda translation: translation.id)
        for trans_input, (input_idx, chunks) in zip(trans_inputs, chunks_by_input_idx):
            chunks = list(chunks)  # type: ignore
            if len(chunks) == 1:  # type: ignore
                translation = chunks[0].translation  # type: ignore
            else:
                translations_to_concat = [translated_chunk.translation for translated_chunk in chunks]
                translation = self._concat_translations(translations_to_concat)

            results.append(self._make_result(trans_input, translation))

        return results

    def _get_inference_input(self, trans_inputs: List[TranslatorInput]) -> Tuple[mx.nd.NDArray, int]:
        """
        Returns NDArray of source ids (shape=(batch_size, bucket_key, num_factors)) and corresponding bucket_key.
        Also checks correctness of translator inputs.

        :param trans_inputs: List of TranslatorInputs.
        :return NDArray of source ids and bucket key.
        """
        bucket_key = data_io.get_bucket(max(len(inp.tokens) for inp in trans_inputs), self.buckets_source)

        source = mx.nd.zeros((len(trans_inputs), bucket_key, self.num_source_factors), ctx=self.context)

        for j, trans_input in enumerate(trans_inputs):
            num_tokens = len(trans_input)
            source[j, :num_tokens, 0] = data_io.tokens2ids(trans_input.tokens, self.source_vocabs[0])

            factors = trans_input.factors if trans_input.factors is not None else []
            num_factors = 1 + len(factors)
            if num_factors != self.num_source_factors:
                logger.warning("Input %d factors, but model(s) expect %d", num_factors,
                               self.num_source_factors)
            for i, factor in enumerate(factors[:self.num_source_factors - 1], start=1):
                # fill in as many factors as there are tokens
                source[j, :num_tokens, i] = data_io.tokens2ids(factor, self.source_vocabs[i])[:num_tokens]

        return source, bucket_key

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrix, and score.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation + attention and score.
        :return: TranslatorOutput.
        """
        # remove special sentence start symbol (<s>) from the output:
        target_ids = translation.target_ids[1:]
        attention_matrix = translation.attention_matrix[1:, :]

        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]
        target_string = C.TOKEN_SEPARATOR.join(
            target_token for target_id, target_token in zip(target_ids, target_tokens) if
            target_id not in self.stop_ids)
        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]

        if isinstance(translation.beam_history, list):
            beam_histories = translation.beam_history
        else:
            beam_histories = [translation.beam_history]

        return TranslatorOutput(id=trans_input.sentence_id,
                                translation=target_string,
                                tokens=target_tokens,
                                attention_matrix=attention_matrix,
                                score=translation.score,
                                beam_histories=beam_histories)

    def _concat_translations(self, translations: List[Translation]) -> Translation:
        """
        Combine translations through concatenation.

        :param translations: A list of translations (sequence, attention_matrix), score and length.
        :return: A concatenation if the translations with a score.
        """
        return _concat_translations(translations, self.start_id, self.stop_ids, self.length_penalty)

    def _translate_nd(self,
                      source: mx.nd.NDArray,
                      source_length: int) -> List[Translation]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.

        :return: Sequence of translations.
        """
        return self._get_best_from_beam(*self._beam_search(source, source_length))

    def _encode(self, sources: mx.nd.NDArray, source_length: int) -> List[ModelState]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param sources: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :return: List of ModelStates.
        """
        return [model.run_encoder(sources, source_length) for model in self.models]

    def _decode_step(self,
                     sequences: mx.nd.NDArray,
                     step: int,
                     source_length: int,
                     states: List[ModelState],
                     models_output_layer_w: List[mx.nd.NDArray],
                     models_output_layer_b: List[mx.nd.NDArray]) \
            -> Tuple[mx.nd.NDArray, mx.nd.NDArray, List[ModelState]]:
        """
        Returns decoder predictions (combined from all models), attention scores, and updated states.

        :param sequences: Sequences of current hypotheses. Shape: (batch_size * beam_size, max_output_length).
        :param step: Beam search iteration.
        :param source_length: Length of the input sequence.
        :param states: List of model states.
        :param models_output_layer_w: Custom model weights for logit computation (empty for none).
        :param models_output_layer_b: Custom model biases for logit computation (empty for none).
        :return: (probs, attention scores, list of model states)
        """
        bucket_key = (source_length, step)
        prev_word = sequences[:, step - 1]

        model_probs, model_attention_probs, model_states = [], [], []
        # We use zip_longest here since we'll have empty lists when not using restrict_lexicon
        for model, out_w, out_b, state in itertools.zip_longest(
                self.models, models_output_layer_w, models_output_layer_b, states):
            decoder_outputs, attention_probs, state = model.run_decoder(prev_word, bucket_key, state)
            # Compute logits and softmax with restricted vocabulary
            if self.restrict_lexicon:
                logits = model.output_layer(decoder_outputs, out_w, out_b)
                probs = mx.nd.softmax(logits)
            else:
                # Otherwise decoder outputs are already target vocab probs
                probs = decoder_outputs
            model_probs.append(probs)
            model_attention_probs.append(attention_probs)
            model_states.append(state)
        neg_logprobs, attention_probs = self._combine_predictions(model_probs, model_attention_probs)
        return neg_logprobs, attention_probs, model_states

    def _combine_predictions(self,
                             probs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined negative log probabilities, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            neg_logprobs = -mx.nd.log(probs[0])  # pylint: disable=invalid-unary-operand-type
        else:
            neg_logprobs = self.interpolation_func(probs)
        return neg_logprobs, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: int) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, Optional[List[BeamHistory]]]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key).
        :param source_length: Max source length.
        :return List of lists of word ids, list of attentions, array of accumulated length-normalized
                negative log-probs.
        """
        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(source_length)
        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(source_length) for model in self.models),
                              "Models must agree on encoded sequence length")
        # Maximum output length
        max_output_length = self.models[0].get_max_output_length(source_length)

        # General data structure: each row has batch_size * beam blocks for the 1st sentence, with a full beam,
        # then the next block for the 2nd sentence and so on

        # sequences: (batch_size * beam_size, output_length), pre-filled with <s> symbols on index 0
        sequences = mx.nd.full((self.batch_size * self.beam_size, max_output_length), val=C.PAD_ID, ctx=self.context,
                               dtype='int32')
        sequences[:, 0] = self.start_id

        # Beam history
        if self.store_beam:
            beam_histories = [defaultdict(list) for _ in range(self.batch_size)] # type: Optional[List[BeamHistory]]
        else:
            beam_histories = None

        lengths = mx.nd.ones((self.batch_size * self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # attentions: (batch_size * beam_size, output_length, encoded_source_length)
        attentions = mx.nd.zeros((self.batch_size * self.beam_size, max_output_length, encoded_source_length),
                                 ctx=self.context)

        # best_hyp_indices: row indices of smallest scores (ascending).
        best_hyp_indices = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        # best_word_indices: column indices of smallest scores (ascending).
        best_word_indices = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((self.batch_size * self.beam_size, 1), ctx=self.context)

        # reset all padding distribution cells to np.inf
        self.pad_dist[:] = np.inf

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        models_output_layer_w = list()
        models_output_layer_b = list()
        pad_dist = self.pad_dist
        vocab_slice_ids = None  # type: mx.nd.NDArray
        if self.restrict_lexicon:
            # TODO: See note in method about migrating to pure MXNet when set operations are supported.
            #       We currently convert source to NumPy and target ids back to NDArray.
            source_words = source.split(num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            vocab_slice_ids = mx.nd.array(self.restrict_lexicon.get_trg_ids(source_words.astype("int32").asnumpy()),
                                          ctx=self.context)

            if vocab_slice_ids.shape[0] < self.beam_size + 1:
                # This fixes an edge case for toy models, where the number of vocab ids from the lexicon is
                # smaller than the beam size.
                logger.warning("Padding vocab_slice_ids (%d) with EOS to have at least %d+1 elements to expand",
                               vocab_slice_ids.shape[0], self.beam_size)
                n = self.beam_size - vocab_slice_ids.shape[0] + 1
                vocab_slice_ids = mx.nd.concat(vocab_slice_ids,
                                               mx.nd.full((n,), val=self.vocab_target[C.EOS_SYMBOL], ctx=self.context),
                                               dim=0)

            pad_dist = mx.nd.full((self.batch_size * self.beam_size, vocab_slice_ids.shape[0]),
                                  val=np.inf, ctx=self.context)
            for m in self.models:
                models_output_layer_w.append(m.output_layer_w.take(vocab_slice_ids))
                models_output_layer_b.append(m.output_layer_b.take(vocab_slice_ids))

        # (0) encode source sentence, returns a list
        model_states = self._encode(source, source_length)

        for t in range(1, max_output_length):

            # (1) obtain next predictions and advance models' state
            # scores: (batch_size * beam_size, target_vocab_size)
            # attention_scores: (batch_size * beam_size, bucket_key)
            scores, attention_scores, model_states = self._decode_step(sequences,
                                                                       t,
                                                                       source_length,
                                                                       model_states,
                                                                       models_output_layer_w,
                                                                       models_output_layer_b)

            # (2) compute length-normalized accumulated scores in place
            if t == 1 and self.batch_size == 1:  # only one hypothesis at t==1
                scores = scores[:1] / self.length_penalty(lengths[:1])
            else:
                # renormalize scores by length ...
                scores = (scores + scores_accumulated * self.length_penalty(lengths - 1)) / self.length_penalty(lengths)
                # ... but not for finished hyps.
                # their predicted distribution is set to their accumulated scores at C.PAD_ID.
                pad_dist[:, C.PAD_ID] = scores_accumulated[:, 0]
                # this is equivalent to doing this in numpy:
                #   pad_dist[finished, :] = np.inf
                #   pad_dist[finished, C.PAD_ID] = scores_accumulated[finished]
                scores = mx.nd.where(finished, pad_dist, scores)

            # (3) get beam_size winning hypotheses for each sentence block separately
            # TODO(fhieber): once mx.nd.topk is sped-up no numpy conversion necessary anymore.
            scores = scores.asnumpy()  # convert to numpy once to minimize cross-device copying
            for sent in range(self.batch_size):
                rows = slice(sent * self.beam_size, (sent + 1) * self.beam_size)
                sliced_scores = scores if t == 1 and self.batch_size == 1 else scores[rows]
                # TODO we could save some tiny amount of time here by not running smallest_k for a finished sent
                (best_hyp_indices[rows], best_word_indices[rows]), \
                scores_accumulated[rows, 0] = utils.smallest_k(sliced_scores, self.beam_size, t == 1)
                # offsetting since the returned smallest_k() indices were slice-relative
                best_hyp_indices[rows] += rows.start

            # Map from restricted to full vocab ids if needed
            if self.restrict_lexicon:
                best_word_indices[:] = vocab_slice_ids.take(best_word_indices)

            # (4) get hypotheses and their properties for beam_size winning hypotheses (ascending)
            sequences = mx.nd.take(sequences, best_hyp_indices)
            lengths = mx.nd.take(lengths, best_hyp_indices)
            finished = mx.nd.take(finished, best_hyp_indices)
            attention_scores = mx.nd.take(attention_scores, best_hyp_indices)
            attentions = mx.nd.take(attentions, best_hyp_indices)

            # (5) update best hypotheses, their attention lists and lengths (only for non-finished hyps)
            # pylint: disable=unsupported-assignment-operation
            sequences[:, t] = best_word_indices
            attentions[:, t, :] = attention_scores
            lengths += mx.nd.cast(1 - mx.nd.expand_dims(finished, axis=1), dtype='float32')


            # (6) optionally save beam history
            if self.store_beam:
                unnormalized_scores = scores_accumulated * self.length_penalty(lengths -1)
                for sent in range(self.batch_size):
                    rows = slice(sent * self.beam_size, (sent + 1) * self.beam_size)

                    best_word_indices_sent = best_word_indices[rows].asnumpy().tolist()
                    # avoid adding columns for finished sentences
                    if any(x for x in best_word_indices_sent if x != C.PAD_ID):
                        beam_histories[sent]["predicted_ids"].append(best_word_indices_sent)
                        beam_histories[sent]["predicted_tokens"].append([self.vocab_target_inv[x] for x in
                                                                    best_word_indices_sent])
                        # for later sentences in the matrix, shift from e.g. [5, 6, 7, 8, 6] to [0, 1, 3, 4, 1]
                        shifted_parents = best_hyp_indices[rows] - (sent * self.beam_size)
                        beam_histories[sent]["parent_ids"].append(shifted_parents.asnumpy().tolist())

                        beam_histories[sent]["scores"].append(unnormalized_scores[rows].asnumpy().flatten().tolist())
                        beam_histories[sent]["normalized_scores"].append(scores_accumulated[rows].asnumpy().flatten().tolist())


            # (7) determine which hypotheses in the beam are now finished
            finished = ((best_word_indices == C.PAD_ID) + (best_word_indices == self.vocab_target[C.EOS_SYMBOL]))
            if mx.nd.sum(finished).asscalar() == self.batch_size * self.beam_size:  # all finished
                break

            # (8) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices)

        return sequences, attentions, scores_accumulated, lengths, beam_histories

    def _get_best_from_beam(self,
                            sequences: mx.nd.NDArray,
                            attention_lists: mx.nd.NDArray,
                            accumulated_scores: mx.nd.NDArray,
                            lengths: mx.nd.NDArray,
                            beam_histories: Optional[List[BeamHistory]]) -> List[Translation]:
        """
        Return the best (aka top) entry from the n-best list.

        :param sequences: Array of word ids. Shape: (batch_size * beam_size, bucket_key).
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch_size * self.beam_size, max_output_length, encoded_source_length).
        :param accumulated_scores: Array of length-normalized negative log-probs.
        :return: Top sequence, top attention matrix, top accumulated score (length-normalized
                 negative log-probs) and length.
        """
        utils.check_condition(sequences.shape[0] == attention_lists.shape[0] \
                              == accumulated_scores.shape[0] == lengths.shape[0], "Shape mismatch")
        # sequences & accumulated scores are in latest 'k-best order', thus 0th element is best
        best = 0
        result = []
        for sent in range(self.batch_size):
            idx = sent * self.beam_size + best
            length = int(lengths[idx].asscalar())
            sequence = sequences[idx][:length].asnumpy().tolist()
            # attention_matrix: (target_seq_len, source_seq_len)
            attention_matrix = np.stack(attention_lists[idx].asnumpy()[:length, :], axis=0)
            score = accumulated_scores[idx].asscalar()
            if beam_histories is not None:
                history = beam_histories[sent]
                result.append(Translation(sequence, attention_matrix, score, [history]))
            else:
                result.append(Translation(sequence, attention_matrix, score))
        return result
