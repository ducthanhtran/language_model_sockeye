from sockeye.config import Config
from sockeye.rnn import RNNConfig
from sockeye.encoder import EmbeddingConfig
from sockeye.loss import LossConfig

LANGUAGE_MODEL_PREFIX = "lm_"

class LanguageModelConfig(Config):
    """
    Defines the configuration for our stacked RNN language model.
    """
    def __init__(self,
                 max_seq_len: int,
                 vocab_size: int,
                 num_embed: int,
                 rnn_config: RNNConfig,
                 config_embed: EmbeddingConfig,
                 config_loss: LossConfig) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.rnn_config = rnn_config
        self.config_embed = config_embed
        self.config_loss = config_loss
