import sys

sys.path.append('../')

from .lm_data_io import LMDataConfig

from sockeye.config import Config
from sockeye.rnn import RNNConfig
from sockeye.encoder import EmbeddingConfig
from sockeye.loss import LossConfig

JSON_SUFFIX = ".json"
LM_PREFIX = "lm_"
LM_VOCAB_NAME = "vocab.lm" + JSON_SUFFIX
LM_DATA_INFO = LM_PREFIX + "data.info"


class LMConfig(Config):
    """
    Defines the configuration for our stacked RNN language model.
    """

    def __init__(self,
                 config_data: LMDataConfig,
                 target_vocab_size: int,
                 num_embed_target: int,
                 rnn_config: RNNConfig,
                 config_embed: EmbeddingConfig,
                 config_loss: LossConfig,
                 weight_tying: bool = False) -> None:
        super().__init__()
        self.max_seq_len_target = config_data.max_seq_len_target
        self.target_vocab_size = target_vocab_size
        self.num_embed_target = num_embed_target
        self.rnn_config = rnn_config
        self.config_embed = config_embed
        self.config_loss = config_loss
        self.weight_tying = weight_tying
