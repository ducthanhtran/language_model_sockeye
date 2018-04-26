from . import lm_common
from typing import NamedTuple
from sockeye.decoder import Decoder
from sockeye.rnn import get_stacked_rnn

import mxnet as mx


RecurrentDecoderState = NamedTuple('RecurrentDecoderState', [
    ('hidden', mx.sym.Symbol),
    ('layer_states', List[mx.sym.Symbol]),
])


class LanguageModelDecoder(Decoder):

    def __init__(self,
                 lm_config: lm_common.LanguageModelConfig,
                 prefix: str = lm_common.LM_PREFIX) -> None:
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
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        prev_hidden, *layer_states = states

        prev_state = RecurrentDecoderState(prev_hidden, list(layer_states))

        # state.hidden: (batch_size, rnn_num_hidden)
        state = self._step(target_embed_prev, prev_state)
        return state.hidden, [state.hidden, state.layer_states]

    def get_initial_state(self,
                          target_embed_lengths: mx.sym.Symbol) -> RecurrentDecoderState:
        # For the moment we utilize zero vectors.
        # Infer batch size from target embed lengths.
        zeros = mx.sym.expand_dims(mx.sym.zeros_like(target_embed_lengths), axis=1)
        hidden = mx.sym.tile(data=zeros, reps=(1, self.num_hidden))

        initial_layer_states = []
        for state_idx, (_, init_num_hidden) in enumerate(sum([rnn.state_shape for rnn in self.get_rnn_cells()], [])):
            init = mx.sym.tile(data=zeros, reps=(1, init_num_hidden))
            initial_layer_states.append(init)
        return RecurrentDecoderState(hidden, layer_states)

    def init_states(self,
                    target_embed_lengths: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        # Used in inference phase.
        hidden, layer_states = self.get_initial_state(target_embed_lengths)
        return [hidden] + layer_states

    def reset(self):
        self.stacked_rnn.reset()
        cells_to_reset = list(self.stacked_rnn._cells) # Shallow copy of cells
        for cell in cells_to_reset:
            # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
            if isinstance(cell, mx.rnn.ModifierCell):
                cell.base_cell.reset()
            cell.reset()

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        return [self.stacked_rnn]

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
