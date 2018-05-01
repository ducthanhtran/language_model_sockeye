import logging
import sys
from typing import Any, Iterable, List, Tuple, Optional, Sized

sys.path.append('../')

import mxnet as mx
import numpy as np

from sockeye import config
from sockeye import constants as C
from sockeye import data_io
from sockeye import vocab
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class MonolingualDataSet(Sized):
    """
    Bucketed monolingual data set with labels

    target: given monolingual sent + BOS
    label: given monolingual sent + EOS
    """

    def __init__(self,
                 target: List[mx.nd.array],
                 label: List[mx.nd.array]) -> None:
        check_condition(len(target) == len(label),
                        "Number of buckets for target/label do not match: %d/%d." % (len(target),
                                                                                     len(label)))
        self.target = target
        self.label = label

    def __len__(self) -> int:
        return len(self.target)

    def get_bucket_counts(self):
        return [len(self.target[buck_idx]) for buck_idx in range(len(self))]

    def save(self, fname: str):
        """
        Saves the dataset to a binary .npy file.
        """
        mx.nd.save(fname, self.target + self.label)

    @staticmethod
    def load(fname: str) -> 'MonolingualDataSet':
        """
        Loads a dataset from a binary .npy file.
        """
        data = mx.nd.load(fname)
        n = len(data) // 2
        target = data[:n]
        label = data[n:]
        assert len(target) == len(label)
        return MonolingualDataSet(target, label)

    def fill_up(self,
                bucket_batch_sizes: List[data_io.BucketBatchSize],
                fill_up: str,
                seed: int = 42) -> 'MonolingualDataSet':
        """
        Returns a new dataset with buckets filled up using the specified fill-up strategy.

        :param bucket_batch_sizes: Bucket batch sizes.
        :param fill_up: Fill-up strategy.
        :param seed: The random seed used for sampling sentences to fill up.
        :return: New dataset with buckets filled up to the next multiple of batch size
        """
        target = list(self.target)
        label = list(self.label)

        rs = np.random.RandomState(seed)

        for bucket_idx in range(len(self)):
            bucket = bucket_batch_sizes[bucket_idx].bucket
            bucket_batch_size = bucket_batch_sizes[bucket_idx].batch_size
            bucket_target = self.target[bucket_idx]
            bucket_label = self.label[bucket_idx]
            num_samples = bucket_target.shape[0]

            if num_samples % bucket_batch_size != 0:
                if fill_up == 'replicate':
                    rest = bucket_batch_size - num_samples % bucket_batch_size
                    logger.info("Replicating %d random samples from %d samples in bucket %s "
                                "to size it to multiple of %d",
                                rest, num_samples, bucket, bucket_batch_size)
                    random_indices = mx.nd.array(rs.randint(num_samples, size=rest))
                    target[bucket_idx] = mx.nd.concat(bucket_target, bucket_target.take(random_indices), dim=0)
                    label[bucket_idx] = mx.nd.concat(bucket_label, bucket_label.take(random_indices), dim=0)
                else:
                    raise NotImplementedError('Unknown fill-up strategy')

        return MonolingualDataSet(target, label)

    def permute(self, permutations: List[mx.nd.NDArray]) -> 'MonolingualDataSet':
        assert len(self) == len(permutations)
        target = []
        label = []
        for buck_idx in range(len(self)):
            num_samples = self.target[buck_idx].shape[0]
            if num_samples:  # not empty bucket
                permutation = permutations[buck_idx]
                target.append(self.target[buck_idx].take(permutation))
                label.append(self.label[buck_idx].take(permutation))
            else:
                target.append(self.target[buck_idx])
                label.append(self.label[buck_idx])

        return MonolingualDataSet(target, label)


def get_monolingual_bucket(buckets: List[int],
                           length_target: int) -> Optional[Tuple[int, int]]:
    """
    Returns bucket index and bucket from a list of buckets, given target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket = None, None  # type: Tuple[int, int]
    for j, target_bkt in enumerate(buckets):
        if target_bkt >= length_target:
            bucket = j, target_bkt
            break
    return bucket


class RawMonolingualDatasetLoader:
    """
    Loads a data set of variable-length monolingual sequences into buckets of NDArrays.

    :param buckets: Bucket list.
    :param eos_id: End-of-sentence id.
    :param pad_id: Padding id.
    :param eos_id: Unknown id.
    :param dtype: Data type.
    """

    def __init__(self,
                 buckets: List[int],
                 eos_id: int,
                 pad_id: int,
                 dtype: str = 'float32') -> None:
        self.buckets = buckets
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.dtype = dtype

    def load(self,
             target_sentences: Iterable[List[Any]],
             num_samples_per_bucket: List[int]) -> 'MonolingualDataSet':

        assert len(num_samples_per_bucket) == len(self.buckets)
        data_target = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                       for target_len, num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_label = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                      for target_len, num_samples in zip(self.buckets, num_samples_per_bucket)]

        bucket_sample_index = [0 for buck in self.buckets]

        # track amount of padding introduced through bucketing
        num_tokens_target = 0
        num_pad_target = 0

        # Bucket sentences as padded np arrays
        for target in target_sentences:
            target_len = len(target)
            buck_index, buck = get_monolingual_bucket(self.buckets, target_len)
            if buck is None:
                continue  # skip this sentence

            num_tokens_target += buck
            num_pad_target += buck - target_len

            sample_index = bucket_sample_index[buck_index]
            # NOTE(yunsukim86): target already contains BOS
            data_target[buck_index][sample_index, :target_len] = target
            # NOTE(fhieber): while this is wasteful w.r.t memory, we need to explicitly create the label sequence
            # with the EOS symbol here sentence-wise and not per-batch due to variable sequence length within a batch.
            # Once MXNet allows item assignments given a list of indices (probably MXNet 1.0): e.g a[[0,1,5,2]] = x,
            # we can try again to compute the label sequence on the fly in next().
            # NOTE(yunsukim86): BOS dropped for label
            data_label[buck_index][sample_index, :target_len] = target[1:] + [self.eos_id]

            bucket_sample_index[buck_index] += 1

        for i in range(len(data_target)):
            data_target[i] = mx.nd.array(data_target[i], dtype=self.dtype)
            data_label[i] = mx.nd.array(data_label[i], dtype=self.dtype)

        if num_tokens_target > 0:
            logger.info("Created bucketed monolingual data set. Introduced padding: target=%.1f%%)",
                        num_pad_target / num_tokens_target * 100)

        return MonolingualDataSet(data_target, data_label)


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

    validation_data_statistics = data_io.get_data_statistics([validation_input_sentences],
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
    logger.info("[LM] Creating training data iterator")
    logger.info("===============================")

    # Pass 1: Length ratio is always 1
    length_ratio_mean = 1
    length_ratio_std = 0

    # Define buckets
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, bucket_width,
                                      length_ratio_mean) if bucketing else [
        (max_seq_len, max_seq_len)]

    # Input starts from <s>
    input_sentences = data_io.SequenceReader(train_data, vocab)
    output_sentences = data_io.SequenceReader(train_data, vocab, add_bos=True)

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

    validation_iter = lm_get_validation_data_iter(data_loader=data_loader,
                                                  validation_data=validation_data,
                                                  buckets=buckets,
                                                  bucket_batch_sizes=bucket_batch_sizes,
                                                  vocab=vocab,
                                                  max_seq_len=max_seq_len,
                                                  batch_size=batch_size,
                                                  fill_up=fill_up)

    return train_iter, validation_iter, config_data, data_info
