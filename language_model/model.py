from sockeye.config import Config
from sockeye.rnn import RNNConfig, get_stacked_rnn
from sockeye.decoder import Decoder


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


class LanguageModel(Decoder):

    def __init__(self,
                 lm_config: LanguageModelConfig,
                 prefix: str = LANGUAGE_MODEL_PREFIX) -> None:
        self.lm_config = lm_config
        self.rnn_config = self.lm_config.rnn_config
        self.prefix = prefix

        # use Sockeye's internal stacked RNN computation graph
        self.stacked_rnn = get_stacked_rnn(config=self.rnn_config, prefix=self.prefix)
