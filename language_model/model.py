# TODO: language model as a computation graph

from sockeye.config import Config

class LanguageModelConfig(Config):
    """
    Defines the configuration for our stacked RNN language model.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_rnn_layers: int,
                 max_seq_len_target: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_rnn_layers = num_rnn_layers
        self.max_seq_len_target = max_seq_len_target
