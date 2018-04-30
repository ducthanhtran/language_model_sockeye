import sys

sys.path.append('../')

from sockeye.config import Config
from sockeye.data_io import DataConfig
from sockeye.rnn import RNNConfig
from sockeye.encoder import EmbeddingConfig
from sockeye.loss import LossConfig


JSON_SUFFIX = ".json"
LM_PREFIX = "lm_"
LM_VOCAB_NAME = "vocab.lm" + JSON_SUFFIX
LM_DATA_INFO = LM_PREFIX + "data.info"


class LanguageModelConfig(Config):
    """
    Defines the configuration for our stacked RNN language model.
    """
    def __init__(self,
                 config_data: DataConfig,
                 vocab_size: int,
                 num_embed: int,
                 rnn_config: RNNConfig,
                 config_embed: EmbeddingConfig,
                 config_loss: LossConfig,
                 weight_tying: bool = False) -> None:
        super().__init__()
        self.max_seq_len = config_data.max_seq_len
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.rnn_config = rnn_config
        self.config_embed = config_embed
        self.config_loss = config_loss
        self.weight_tying = weight_tying
