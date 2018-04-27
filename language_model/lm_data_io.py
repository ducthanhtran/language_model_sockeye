import logging
import sys
from typing import List, Tuple, Optional

sys.path.append('../')

from sockeye import config
from sockeye import constants as C
from sockeye import data_io
from sockeye import vocab

logger = logging.getLogger(__name__)


class LanguageModelDataStatistics(config.Config):

    def __init__(self,
                 num_sents: int,
                 num_discarded,
                 num_tokens,
                 num_unks,
                 max_observed_len,
                 size_vocab,
                 buckets: List[Tuple[int, int]],
                 num_sents_per_bucket: List[int],
                 mean_len_per_bucket: List[Optional[float]]) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.num_discarded = num_discarded
        self.num_tokens = num_tokens
        self.num_unks = num_unks
        self.max_observed_len = max_observed_len
        self.size_vocab = size_vocab
        self.buckets = buckets
        self.num_sents_per_bucket = num_sents_per_bucket
        self.average_len_per_bucket = mean_len_per_bucket

    def log(self, bucket_batch_sizes: Optional[List[data_io.BucketBatchSize]] = None):
        logger.info("[LM] Tokens: %d", self.num_tokens)
        if self.num_tokens > 0:
            logger.info("[LM] Vocabulary coverage: %.0f%%",
                        (1 - self.num_unks / self.num_tokens) * 100)
        logger.info("[LM] %d sequences across %d buckets", self.num_sents, len(self.num_sents_per_bucket))
        logger.info("[LM] %d sequences did not fit into buckets and were discarded", self.num_discarded)
        if bucket_batch_sizes is not None:
            data_io.describe_data_and_buckets(self, bucket_batch_sizes)


class LanguageModelDataStatisticsAccumulator:

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 vocab_target: Dict[str, int],
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_target = len(vocab_target)
        self.num_sents = 0
        self.num_discarded = 0
        self.num_tokens_target = 0
        self.num_unks_target = 0
        self.max_observed_len_target = 0
        self._mean_len_target_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence_pair(self,
                      target: List[int],
                      bucket_idx: Optional[int]):
        if bucket_idx is None:
            self.num_discarded += 1
            return

        target_len = len(target)

        self._mean_len_target_per_bucket[bucket_idx].update(target_len)

        self.num_sents += 1
        self.num_tokens_target += target_len
        self.max_observed_len_target = max(target_len, self.max_observed_len_target)

        self.num_unks_target += target.count(self.unk_id_target)

    @property
    def mean_len_target_per_bucket(self) -> List[Optional[float]]:
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None
                for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def statistics(self):
        num_sents_per_bucket = [mean_and_variance.count for mean_and_variance in self._mean_len_target_per_bucket]
        return LanguageModelDataStatistics(num_sents=self.num_sents,
                                           num_discarded=self.num_discarded,
                                           num_tokens_target=self.num_tokens_target,
                                           num_unks_target=self.num_unks_target,
                                           max_observed_len_target=self.max_observed_len_target,
                                           size_vocab_target=self.size_vocab_target,
                                           length_ratio_mean=self.length_ratio_mean,
                                           length_ratio_std=self.length_ratio_std,
                                           buckets=self.buckets,
                                           num_sents_per_bucket=num_sents_per_bucket,
                                           mean_len_target_per_bucket=self.mean_len_target_per_bucket)


class LanguageModelDataInfo(config.Config):
    """
    Stores training data information that is not relevant for inference.
    """

    def __init__(self,
                 data: str,
                 vocab: Optional[str],
                 num_shards: int) -> None:
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.num_shards = num_shards

class LanguageModelDataConfig(config.Config):
    """
    Stores data statistics relevant for inference.
    """

    def __init__(self,
                 data_statistics: LanguageModelDataStatistics,
                 max_seq_len: int) -> None:
        super().__init__()
        self.data_statistics = data_statistics
        self.max_seq_len = max_seq_len

def lm_get_validation_data_iter(data_loader: data_io.RawParallelDatasetLoader,
                             validation_data: str,
                             buckets: List[Tuple[int, int]],
                             bucket_batch_sizes: List[data_io.BucketBatchSize],
                             vocab: vocab.Vocab,
                             max_seq_len: int,
                             batch_size: int,
                             fill_up: str) -> 'ParallelSampleIter':
    """
    Returns a ParallelSampleIter for the validation data.
    """
    logger.info("=================================")
    logger.info("[LM] Creating validation data iterator")
    logger.info("=================================")

    length_ratio_mean = 1
    length_ratio_std = 0

    validation_input_sentences = data_io.SequenceReader(validation_data, vocab, add_bos=True)
    validation_output_sentences = data_io.SequenceReader(validation_data, vocab, limit=None)

    validation_data_statistics = get_data_statistics([validation_input_sentences],
                                                     validation_output_sentences,
                                                     buckets,
                                                     length_ratio_mean,
                                                     length_ratio_std,
                                                     [vocab], vocab)

    validation_data_statistics.log(bucket_batch_sizes)

    validation_data_loaded = data_loader.load([validation_input_sentences],
                                              validation_output_sentences,
                                              validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    return data_io.ParallelSampleIter(data=validation_data_loaded,
                              buckets=buckets,
                              batch_size=batch_size,
                              bucket_batch_sizes=bucket_batch_sizes,
                              num_factors=1)


def lm_get_training_data_iters(train_data: str,
                               validation_data: str,
                               vocab: vocab.Vocab,
                               vocab_path: Optional[str],
                               batch_size: int,
                               batch_by_words: bool,
                               batch_num_devices: int,
                               fill_up: str,
                               max_seq_len: int,
                               bucketing: bool,
                               bucket_width: int) -> Tuple['BaseParallelSampleIter',
                                                           'BaseParallelSampleIter',
                                                           'DataConfig', 'DataInfo']:
    """
    Returns data iterators for training and validation data.

    :param train_data: Path to training data.
    :param validation_data: Path to validation data.
    :param vocab: Vocabulary.
    :param vocab_path: Path to vocabulary.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len: Maximum sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: Tuple of (training data iterator, validation data iterator, data config, data info).
    """
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")

    # Pass 1: Length ratio is always 1
    length_ratio_mean = 1
    length_ratio_std = 0

    # Define buckets
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, bucket_width,
                                      length_ratio_mean) if bucketing else [
        (max_seq_len, max_seq_len)]

    # Input starts from <s>
    input_sentences = data_io.SequenceReader(train_data, vocab, add_bos=True)
    output_sentences = data_io.SequenceReader(train_data, vocab)

    # Pass 2: Get data statistics (for debugging)
    data_statistics = data_io.get_data_statistics([input_sentences], output_sentences, buckets,
                                          length_ratio_mean, length_ratio_std,
                                          [vocab], vocab)

    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    # </s> is added here in the output side (labels)
    data_loader = data_io.RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    training_data = data_loader.load([input_sentences], output_sentences,
                                         data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    data_info = LanguageModelDataInfo(data=train_data,
                                      vocab=vocab_path,
                                      num_shards=1)

    config_data = LanguageModelDataConfig(data_statistics=data_statistics,
                                          max_seq_len=max_seq_len)

    train_iter = data_io.ParallelSampleIter(data=training_data,
                                    buckets=buckets,
                                    batch_size=batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_factors=1)

    validation_sources = []
    validation_target = ""
    source_vocabs = {}
    max_seq_len_source = 0

    validation_iter = data_io.get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len,
                                               batch_size=batch_size,
                                               fill_up=fill_up)

    return train_iter, validation_iter, config_data, data_info
