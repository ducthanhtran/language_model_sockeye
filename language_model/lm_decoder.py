import sys
import mxnet as mx
from typing import List, NamedTuple, Tuple

from . import lm_common

sys.path.append('../')

import sockeye.constants as C
from sockeye.decoder import Decoder
from sockeye.rnn import get_stacked_rnn


LMDecoderState = NamedTuple('LMDecoderState', [
    ('layer_states', List[mx.sym.Symbol])
])


class LanguageModelDecoder(Decoder):

    def __init__(self,
                 lm_config: lm_common.LMConfig,
                 prefix: str = lm_common.LM_PREFIX) -> None:
        self.lm_config = lm_config
        self.rnn_config = self.lm_config.rnn_config
        self.prefix = prefix

        self.num_hidden = self.rnn_config.num_hidden

        # use Sockeye's internal stacked RNN computation graph
        self.stacked_rnn = get_stacked_rnn(config=self.rnn_config, prefix=self.prefix)
        self.stacked_rnn_state_number = len(self.stacked_rnn.state_shape)


    def decode_sequence(self,
                        target_embed: mx.sym.Symbol,
                        target_embed_lengths: mx.sym.Symbol,
                        target_embed_max_length: int) -> mx.sym.Symbol:
        target_embed = mx.sym.split(data=target_embed, num_outputs=target_embed_max_length, axis=1, squeeze_axis=True)
        state = self.get_initial_state(target_embed_lengths)

        top_hidden_states = []  # type: List[mx.sym.Symbol]
        self.reset()
        # TODO: do we take <s> into account here?
        for seq_idx in range(target_embed_max_length):
            # hidden: (batch_size, rnn_num_hidden)
            top_hidden, state = self._step(target_embed[seq_idx],
                                           state,
                                           seq_idx)
            top_hidden_states.append(top_hidden)

        # concatenate along time axis: (batch_size, target_embed_max_length, rnn_num_hidden)
        return mx.sym.stack(*top_hidden_states, axis=1, name='%stop_hidden_stack' % self.prefix)

    def decode_step(self,
                    step: int,
                    target_embed_prev: mx.sym.Symbol,
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        prev_state = LMDecoderState(list(states))

        # top_hidden: (batch_size, rnn_num_hidden)
        top_hidden, cur_state = self._step(target_embed_prev, prev_state)

        return top_hidden, cur_state.layer_states

    def get_initial_state(self,
                          target_embed_lengths: mx.sym.Symbol) -> LMDecoderState:
        # For the moment we utilize zero vectors.
        # Infer batch size from target embed lengths.
        zeros = mx.sym.expand_dims(mx.sym.zeros_like(target_embed_lengths), axis=1)

        initial_layer_states = []
        for state_idx, (_, init_num_hidden) in enumerate(sum([rnn.state_shape for rnn in self.get_rnn_cells()], [])):
            init = mx.sym.tile(data=zeros, reps=(1, init_num_hidden))
            initial_layer_states.append(init)
        return LMDecoderState(initial_layer_states)

    def init_states(self,
                    target_embed_lengths: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        # Used in inference phase.
        return self.get_initial_state(target_embed_lengths)[0]

    def reset(self):
        self.stacked_rnn.reset()
        cells_to_reset = list(self.stacked_rnn._cells) # Shallow copy of cells
        for cell in cells_to_reset:
            # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
            if isinstance(cell, mx.rnn.ModifierCell):
                cell.base_cell.reset()
            cell.reset()

    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this decoder.
        """
        return self.num_hidden

    def state_variables(self, target_max_length: int) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :param target_max_length: Current target sequence lengths.
        :return: List of symbolic variables.
        """
        return [mx.sym.Variable("%sdec_hidden_%d" % (self.prefix, i)) for i in
                range(len(sum([rnn.state_info for rnn in self.get_rnn_cells()], [])))]

    def state_shapes(self,
                     batch_size: int,
                     target_max_length: int) -> List[mx.io.DataDesc]:
        """
        Returns a list of shape descriptions given batch size.
        Used for inference.

        :param batch_size: Batch size during inference.
        :param target_max_length: Current target sequence length.
        :return: List of shape descriptions.
        """
        return [mx.io.DataDesc("%sdec_hidden_%d" % (self.prefix, i),
                               (batch_size, num_hidden),
                               layout=C.BATCH_MAJOR) for i, (_, num_hidden) in enumerate(
                   sum([rnn.state_shape for rnn in self.get_rnn_cells()], [])
               )]

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        return [self.stacked_rnn]

    def _step(self, target_embed_prev: mx.sym.Symbol,
              state: LMDecoderState,
              seq_idx: int = 0) -> Tuple[mx.sym.Symbol, LMDecoderState]:
        """
        Performs a single RNN step.

        :param target_embed_prev: previous output word as embedded vector
        """

        # Unroll stacked RNN for one timestep
        # rnn_output: (batch_size, rnn_num_hidden)
        # rnn_states: num_layers * [batch_size, rnn_num_hidden]
        top_hidden, layer_states = self.stacked_rnn(target_embed_prev, state.layer_states[:self.stacked_rnn_state_number])
        return top_hidden, LMDecoderState(layer_states)
