from typing import NamedTuple
from sockeye.config import Config
from sockeye.rnn import RNNConfig, get_stacked_rnn
from sockeye.decoder import Decoder, get_initial_state


LANGUAGE_MODEL_PREFIX = "lm_"


class LanguageModelConfig(Config):
    """
    Defines the configuration for our stacked RNN language model.
    """
    def __init__(self,
                 max_seq_len_target: int,
                 rnn_config: RNNConfig) -> None:
        super().__init__()
        self.max_seq_len_target = max_seq_len_target
        self.rnn_config = rnn_config


RecurrentDecoderState = NamedTuple('RecurrentDecoderState', [
    ('hidden', mx.sym.Symbol),
    ('layer_states', List[mx.sym.Symbol]),
])

class LanguageModel(Decoder):

    def __init__(self,
                 lm_config: LanguageModelConfig,
                 prefix: str = LANGUAGE_MODEL_PREFIX) -> None:
        self.lm_config = lm_config
        self.rnn_config = self.lm_config.rnn_config
        self.prefix = prefix

        # use Sockeye's internal stacked RNN computation graph
        self.stacked_rnn = get_stacked_rnn(config=self.rnn_config, prefix=self.prefix)
        self.stacked_rnn_state_number = len(self.stacked_rnn.state_shape)

    def reset(self):
        self.stacked_rnn.reset()
        cells_to_reset = list(self.stacked_rnn._cells) # Shallow copy of cells
        for cell in cells_to_reset:
            # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
            if isinstance(cell, mx.rnn.ModifierCell):
                cell.base_cell.reset()
            cell.reset()

    def _step(self, prev_output_word: mx.sym.Symbol,
              state: RecurrentDecoderState,
              seq_idx: int = 0) -> RecurrentDecoderState:
        """
        Performs a single RNN step.

        :param prev_output_word: previous output word as embedded vector
        """
        # (1) label feedback from previous step is concatenated with hidden states
        concatenated_input = mx.sym.concat(prev_output_word, state.hidden, dim=1,
                                  name="%sconcat_lm_label_feedback_hidden_state_%d" % (self.prefix, seq_idx))

        # (2) unroll stacked RNN for one timestep
        # rnn_output: (batch_size, rnn_num_hidden)
        # rnn_states: num_layers * [batch_size, rnn_num_hidden]
        rnn_output, rnn_states = \
            self.stacked_rnn(concatenated_input, state.layer_states[:self.stacked_rnn_state_number])

        return RecurrentDecoderState(rnn_output, rnn_states)
