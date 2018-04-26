import logging

from sockeye import config
from sockeye import constants as C
from sockeye.data_io import BaseParallelSampleIter, define_parallel_buckets, get_data_statistics, define_bucket_batch_sizes, BucketBatchSize
from sockeye import vocab

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

    def log(self, bucket_batch_sizes: Optional[List[BucketBatchSize]] = None):
        logger.info("[LM] Tokens: %d", self.num_tokens)
        if self.num_tokens > 0:
            logger.info("[LM] Vocabulary coverage: %.0f%%",
                        (1 - self.num_unks / self.num_tokens) * 100)
        logger.info("[LM] %d sequences across %d buckets", self.num_sents, len(self.num_sents_per_bucket))
        logger.info("[LM] %d sequences did not fit into buckets and were discarded", self.num_discarded)
        if bucket_batch_sizes is not None:
            describe_data_and_buckets(self, bucket_batch_sizes)


def lm_describe_data_and_buckets(data_statistics: LanguageModelDataStatistics, bucket_batch_sizes: List[BucketBatchSize]):
    check_condition(len(bucket_batch_sizes) == len(data_statistics.buckets),
                    "[LM] Number of bucket batch sizes (%d) does not match number of buckets in statistics (%d)."
                    % (len(bucket_batch_sizes), len(data_statistics.buckets)))
    for bucket_batch_size, num_seq in zip(bucket_batch_sizes, data_statistics.num_sents_per_bucket):
        if num_seq > 0:
            logger.info("[LM] Bucket %s: %d samples in %d batches of %d, ~%.1f tokens/batch.",
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.average_words_per_batch)


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

def lm_get_validation_data_iter(data_loader: RawParallelDatasetLoader,
                             validation_data: str,
                             buckets: List[Tuple[int, int]],
                             bucket_batch_sizes: List[BucketBatchSize],
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

    validation_input_sentences = SequenceReader(validation_data, vocab, add_bos=True)
    validation_output_sentences = SequenceReader(validation_data, vocab, limit=None)

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

    return ParallelSampleIter(data=validation_data_loaded,
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
    buckets = define_parallel_buckets(max_seq_len, max_seq_len, bucket_width,
                                      length_ratio_mean) if bucketing else [
        (max_seq_len, max_seq_len)]

    # Input starts from <s>
    input_sentences = SequenceReader(train_data, vocab, add_bos=True)
    output_sentences = SequenceReader(train_data, vocab)

    # Pass 2: Get data statistics (for debugging)
    data_statistics = get_data_statistics([input_sentences], output_sentences, buckets,
                                          length_ratio_mean, length_ratio_std,
                                          [vocab], vocab)

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    # </s> is added here in the output side (labels)
    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    train_data_loaded = data_loader.load([input_sentences], output_sentences,
                                         data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    data_info = LanguageModelDataInfo(data=train_data,
                                      vocab=vocab_path,
                                      num_shards=1)

    config_data = LanguageModelDataConfig(data_statistics=data_statistics,
                                          max_seq_len=max_seq_len)

    train_iter = ParallelSampleIter(data=training_data_loaded,
                                    buckets=buckets,
                                    batch_size=batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_factors=1)

    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=target_vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size,
                                               fill_up=fill_up)

    return train_iter, validation_iter, config_data, data_info
