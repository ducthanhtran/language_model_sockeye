from typing import NamedTuple
from sockeye.config import Config
from sockeye.rnn import RNNConfig, get_stacked_rnn
from sockeye.decoder import Decoder, get_initial_state, get_decoder()
from sockeye.encoder import Embedding, EmbeddingConfig
from sockeye.layers import OutputLayer
from sockeye import constants as C


LANGUAGE_MODEL_PREFIX = "lm_"


class LanguageModelConfig(Config):
    """
    Defines the configuration for our stacked RNN language model.
    """
    def __init__(self,
                 max_seq_len: int,
                 vocab_size: int,
                 num_embed: int,
                 rnn_config: RNNConfig
                 embed_config: EmbeddingConfig) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.rnn_config = rnn_config
        self.embed_config = embed_config


RecurrentDecoderState = NamedTuple('RecurrentDecoderState', [
    ('hidden', mx.sym.Symbol),
    ('layer_states', List[mx.sym.Symbol]),
])


class LanguageModelDecoder(Decoder):

    def __init__(self,
                 lm_config: LanguageModelConfig,
                 prefix: str = LANGUAGE_MODEL_PREFIX) -> None:
        self.lm_config = lm_config
        self.rnn_config = self.lm_config.rnn_config
        self.prefix = prefix

        # use Sockeye's internal stacked RNN computation graph
        self.stacked_rnn = get_stacked_rnn(config=self.rnn_config, prefix=self.prefix)
        self.stacked_rnn_state_number = len(self.stacked_rnn.state_shape)


    def decode_sequence(self,
                        target_embed: mx.sym.Symbol,
                        target_embed_lengths: mx.sym.Symbol,
                        target_embed_max_length: int) -> mx.sym.Symbol:
        target_embed = mx.sym.split(data=target_embed, num_outputs=target_embed_max_length, axis=1, squeeze_axis=True)
        state = self.get_initial_state()

        hidden_states = []  # type: List[mx.sym.Symbol]
        self.reset()
        for seq_idx in range(target_embed_max_length):
            # hidden: (batch_size, rnn_num_hidden)
            state = self._step(target_embed[seq_idx],
                               state,
                               seq_idx)
            hidden_states.append(state.hidden)

        # concatenate along time axis: (batch_size, target_embed_max_length, rnn_num_hidden)
        return mx.sym.stack(*hidden_states, axis=1, name='%shidden_stack' % self.prefix)

    def decode_step(self,
                    step: int,
                    target_embed_prev: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        prev_hidden, *layer_states = states

        prev_state = RecurrentDecoderState(prev_hidden, list(layer_states))

        # state.hidden: (batch_size, rnn_num_hidden)
        state = self._step(target_embed_prev, prev_state)
        return state.hidden, [state.hidden, state.layer_states]

    def get_initial_state() -> RecurrentDecoderState:
        # TODO

    def reset(self):
        self.stacked_rnn.reset()
        cells_to_reset = list(self.stacked_rnn._cells) # Shallow copy of cells
        for cell in cells_to_reset:
            # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
            if isinstance(cell, mx.rnn.ModifierCell):
                cell.base_cell.reset()
            cell.reset()

    def _step(self, target_embed_prev: mx.sym.Symbol,
              state: RecurrentDecoderState,
              seq_idx: int = 0) -> RecurrentDecoderState:
        """
        Performs a single RNN step.

        :param target_embed_prev: previous output word as embedded vector
        """
        # (1) label feedback from previous step is concatenated with hidden states
        concatenated_input = mx.sym.concat(target_embed_prev, state.hidden, dim=1,
                                  name="%sconcat_lm_label_feedback_hidden_state_%d" % (self.prefix, seq_idx))

        # (2) unroll stacked RNN for one timestep
        # rnn_output: (batch_size, rnn_num_hidden)
        # rnn_states: num_layers * [batch_size, rnn_num_hidden]
        rnn_output, rnn_states = \
            self.stacked_rnn(concatenated_input, state.layer_states[:self.stacked_rnn_state_number])

        return RecurrentDecoderState(rnn_output, rnn_states)


class LanguageModel:
    """
    LanguageModel shares components needed for both training and inference.
    The main components of a LanguageModel are
    1) Embedding
    2) Decoder
    3) Output layer

    LanguageModel conatins parameters and their values that are fixed at training time and must be re-used at inference time.

    :param config: Model configuration.
    """

    def __init__(self, config: LanguageModelConfig) -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()
        logger.info("%s", self.config)

        # decoder first (to know the decoder depth)
        self.decoder = LanguageModelDecoder(config=self.config,
                                            prefix=LANGUAGE_MODEL_PREFIX + "decoder_")

        # embedding
        embed_weight, out_weight = self._get_embed_weights()
        self.embedding = Embedding(self.config.embed_config,
                                   prefix=LANGUAGE_MODEL_PREFIX + "embed_",
                                   embed_weight=embed_weight)

        # output layer
        self.output_layer = OutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                        vocab_size=self.config.vocab_size,
                                        weight=out_weight,
                                        weight_normalization=False)

        self.params = None  # type: Optional[Dict]
        self.aux_params = None  # type: Optional[Dict]

    def save_config(self, folder: str):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, LANGUAGE_MODEL_PREFIX + C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> LanguageModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = LanguageModelConfig.load(fname)
        logger.info('ModelConfig loaded from "%s"', fname)
        return cast(LanguageModelConfig, config)  # type: ignore

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        if self.aux_params is not None:
            utils.save_params(self.params.copy(), fname, self.aux_params.copy())
        else:
            utils.save_params(self.params.copy(), fname)
        logging.info('Saved LM params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        utils.check_condition(os.path.exists(fname), "No LM parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.params, self.aux_params = utils.load_params(fname)
        logger.info('Loaded LM params from "%s"', fname)

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, LANGUAGE_MODEL_PREFIX + C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embed_weights(self) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Returns embedding parameters.

        :return: Tuple of parameter symbols.
        """
        w_embed = mx.sym.Variable(LANGUAGE_MODEL_PREFIX + "embed_weight",
                                  shape=(self.config.vocab_size, self.config.num_embed))
        w_out = mx.sym.Variable(LANGUAGE_MODEL_PREFIX + "output_weight",
                                shape=(self.config.vocab_size, self.decoder.get_num_hidden()))

        if self.config.weight_tying:
            logger.info("Tying the LM embeddings and output layer parameters.")
            utils.check_condition(self.config.num_embed == self.decoder.get_num_hidden(),
                                  "Weight tying requires LM embedding size and LM decoder hidden size " +
                                  "to be equal: %d vs. %d" % (self.config.num_embed,
                                                              self.decoder.get_num_hidden()))
            w_out = w_embed

        return w_embed, w_out
