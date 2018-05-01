import logging
import math
import random
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple, Optional, Sized

import mxnet as mx
import numpy as np

sys.path.append('../')

from sockeye import config
from sockeye import constants as C
from sockeye import data_io
from sockeye import vocab
from sockeye.utils import check_condition, OnlineMeanAndVariance

logger = logging.getLogger(__name__)


class MonolingualBucketBatchSize:
    """
    :param bucket: The corresponding bucket.
    :param batch_size: Number of sequences in each batch.
    :param average_words_per_batch: Approximate number of non-padding tokens in each batch.
    """

    def __init__(self, bucket: int, batch_size: int, average_words_per_batch: float) -> None:
        self.bucket = bucket
        self.batch_size = batch_size
        self.average_words_per_batch = average_words_per_batch


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

    # Number of buckets
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
                bucket_batch_sizes: List[MonolingualBucketBatchSize],
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


def get_monolingual_default_bucket_key(buckets: List[int]) -> int:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

class BaseMonolingualSampleIter(mx.io.DataIter, ABC):
    """
    Base monolingual sample iterator.
    """

    def __init__(self,
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 target_data_name,
                 label_name,
                 dtype='float32') -> None:
        super().__init__(batch_size=batch_size)

        self.buckets = list(buckets)
        self.default_bucket_key = get_monolingual_default_bucket_key(self.buckets)
        self.bucket_batch_sizes = bucket_batch_sizes
        self.target_data_name = target_data_name
        self.label_name = label_name
        self.dtype = dtype

        # "Staging area" that needs to fit any size batch we're using by total number of elements.
        # When computing per-bucket batch sizes, we guarantee that the default bucket will have the
        # largest total batch size.
        # Note: this guarantees memory sharing for input data and is generally a good heuristic for
        # other parts of the model, but it is possible that some architectures will have intermediate
        # operations that produce shapes larger than the default bucket size.  In these cases, MXNet
        # will silently allocate additional memory.
        self.provide_data = [
            mx.io.DataDesc(name=self.target_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key),
                           layout=C.BATCH_MAJOR)]
        self.provide_label = [
            mx.io.DataDesc(name=self.label_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key),
                           layout=C.BATCH_MAJOR)]

        self.data_names = [self.target_data_name]
        self.label_names = [self.label_name]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def iter_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> mx.io.DataBatch:
        pass

    @abstractmethod
    def save_state(self, fname: str):
        pass

    @abstractmethod
    def load_state(self, fname: str):
        pass


def get_monolingual_batch_indices(data: MonolingualDataSet,
                                  bucket_batch_sizes: List[MonolingualBucketBatchSize]) -> List[Tuple[int, int]]:
    """
    Returns a list of index tuples that index into the bucket and the start index inside a bucket given
    the batch size for a bucket. These indices are valid for the given dataset.

    :param data: Data to create indices for.
    :param bucket_batch_sizes: Bucket batch sizes.
    :return: List of 2d indices.
    """
    # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
    idxs = []  # type: List[Tuple[int, int]]
    for buck_idx, buck in enumerate(data.target):
        bucket = bucket_batch_sizes[buck_idx].bucket
        batch_size = bucket_batch_sizes[buck_idx].batch_size
        num_samples = data.target[buck_idx].shape[0]
        rest = num_samples % batch_size
        if rest > 0:
            logger.info("Ignoring %d samples from bucket %s with %d samples due to incomplete batch",
                        rest, bucket, num_samples)
        idxs.extend([(buck_idx, j) for j in range(0, num_samples - batch_size + 1, batch_size)])
    return idxs


class MonolingualSampleIter(BaseMonolingualSampleIter):
    """
    Data iterator on a bucketed MonolingualDataSet. Shuffles data at every reset and supports saving and loading the
    iterator state.
    """

    def __init__(self,
                 data: MonolingualDataSet,
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 dtype='float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         target_data_name=target_data_name, label_name=label_name, dtype=dtype)

        # create independent lists to be shuffled
        self.data = MonolingualDataSet(list(data.target), list(data.label))

        # create index tuples (buck_idx, batch_start_pos) into buckets. These will be shuffled.
        self.batch_indices = get_monolingual_batch_indices(self.data, bucket_batch_sizes)
        self.curr_batch_index = 0

        self.inverse_data_permutations = [mx.nd.arange(0, max(1, self.data.target[i].shape[0]))
                                          for i in range(len(self.data))]
        self.data_permutations = [mx.nd.arange(0, max(1, self.data.target[i].shape[0]))
                                  for i in range(len(self.data))]

        self.reset()

    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_batch_index = 0
        # shuffle batch start indices
        random.shuffle(self.batch_indices)

        # restore
        self.data = self.data.permute(self.inverse_data_permutations)

        self.data_permutations, self.inverse_data_permutations = data_io.get_permutations(self.data.get_bucket_counts())

        self.data = self.data.permute(self.data_permutations)

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_batch_index != len(self.batch_indices)

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1

        batch_size = self.bucket_batch_sizes[i].batch_size
        target = self.data.target[i][j:j + batch_size]
        data = [target]
        label = [self.data.label[i][j:j + batch_size]]

        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        # TODO: num pad examples is not set here if fillup strategy would be padding
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    def save_state(self, fname: str):
        """
        Saves the current state of iterator to a file, so that iteration can be
        continued. Note that the data is not saved, i.e. the iterator must be
        initialized with the same parameters as in the first call.

        :param fname: File name to save the information to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self.batch_indices, fp)
            pickle.dump(self.curr_batch_index, fp)
            np.save(fp, [a.asnumpy() for a in self.inverse_data_permutations])
            np.save(fp, [a.asnumpy() for a in self.data_permutations])

    def load_state(self, fname: str):
        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """

        # restore order
        self.data = self.data.permute(self.inverse_data_permutations)

        with open(fname, "rb") as fp:
            self.batch_indices = pickle.load(fp)
            self.curr_batch_index = pickle.load(fp)
            inverse_data_permutations = np.load(fp)
            data_permutations = np.load(fp)

        # Because of how checkpointing is done (pre-fetching the next batch in
        # each iteration), curr_idx should always be >= 1
        assert self.curr_batch_index >= 1
        # Right after loading the iterator state, next() should be called
        self.curr_batch_index -= 1

        # load previous permutations
        self.inverse_data_permutations = []
        self.data_permutations = []

        for bucket in range(len(self.data)):
            inverse_permutation = mx.nd.array(inverse_data_permutations[bucket])
            self.inverse_data_permutations.append(inverse_permutation)

            permutation = mx.nd.array(data_permutations[bucket])
            self.data_permutations.append(permutation)

        self.data = self.data.permute(self.data_permutations)


class LMDataStatistics(config.Config):

    def __init__(self,
                 num_sents: int,
                 num_discarded,
                 num_tokens_target,
                 num_unks_target,
                 max_observed_len_target,
                 size_vocab_target,
                 buckets: List[int],
                 num_sents_per_bucket: List[int],
                 mean_len_target_per_bucket: List[Optional[float]]) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.num_discarded = num_discarded
        self.num_tokens_target = num_tokens_target
        self.num_unks_target = num_unks_target
        self.max_observed_len_target = max_observed_len_target
        self.size_vocab_target = size_vocab_target
        self.buckets = buckets
        self.num_sents_per_bucket = num_sents_per_bucket
        self.average_len_target_per_bucket = mean_len_target_per_bucket

    def log(self, bucket_batch_sizes: Optional[List[MonolingualBucketBatchSize]] = None):
        logger.info("[LM] Tokens: %d", self.num_tokens_target)
        if self.num_tokens_target > 0:
            logger.info("[LM] Vocabulary coverage: %.0f%%",
                        (1 - self.num_unks_target / self.num_tokens_target) * 100)
        logger.info("[LM] %d sequences across %d buckets", self.num_sents, len(self.num_sents_per_bucket))
        logger.info("[LM] %d sequences did not fit into buckets and were discarded", self.num_discarded)
        if bucket_batch_sizes is not None:
            lm_describe_data_and_buckets(self, bucket_batch_sizes)


def lm_describe_data_and_buckets(data_statistics: LMDataStatistics, bucket_batch_sizes: List[MonolingualBucketBatchSize]):
    """
    Describes statistics across buckets
    """
    check_condition(len(bucket_batch_sizes) == len(data_statistics.buckets),
                    "Number of bucket batch sizes (%d) does not match number of buckets in statistics (%d)."
                    % (len(bucket_batch_sizes), len(data_statistics.buckets)))
    for bucket_batch_size, num_seq in zip(bucket_batch_sizes, data_statistics.num_sents_per_bucket):
        if num_seq > 0:
            logger.info("Bucket %s: %d samples in %d batches of %d, ~%.1f tokens/batch.",
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.average_words_per_batch)


class LMDataStatisticsAccumulator:

    def __init__(self,
                 buckets: List[int],
                 vocab_target: Dict[str, int]) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_target = len(vocab_target)
        self.num_sents = 0
        self.num_discarded = 0
        self.num_tokens_target = 0
        self.num_unks_target = 0
        self.max_observed_len_target = 0
        self._mean_len_target_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence(self,
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
        return LMDataStatistics(num_sents=self.num_sents,
                                num_discarded=self.num_discarded,
                                num_tokens_target=self.num_tokens_target,
                                num_unks_target=self.num_unks_target,
                                max_observed_len_target=self.max_observed_len_target,
                                size_vocab_target=self.size_vocab_target,
                                buckets=self.buckets,
                                num_sents_per_bucket=num_sents_per_bucket,
                                mean_len_target_per_bucket=self.mean_len_target_per_bucket)


def lm_get_data_statistics(target_sentences: Iterable[List[int]],
                           buckets: List[int],
                           target_vocab: vocab.Vocab) -> 'LMDataStatistics':
    data_stats_accumulator = LMDataStatisticsAccumulator(buckets, target_vocab)

    for target in target_sentences:
        buck_idx, buck = get_monolingual_bucket(buckets, len(target))
        data_stats_accumulator.sequence(target, buck_idx)

    return data_stats_accumulator.statistics


class LMDataInfo(config.Config):
    """
    Stores training data information that is not relevant for inference.
    """

    def __init__(self,
                 target: str,
                 target_vocab: Optional[str]) -> None:
        super().__init__()
        self.target = target
        self.target_vocab = target_vocab

class LMDataConfig(config.Config):
    """
    Stores data statistics relevant for inference.
    """

    def __init__(self,
                 data_statistics: LMDataStatistics,
                 max_seq_len_target: int) -> None:
        super().__init__()
        self.data_statistics = data_statistics
        self.max_seq_len_target = max_seq_len_target


def lm_get_validation_data_iter(data_loader: RawMonolingualDatasetLoader,
                                validation_target: str,
                                buckets: List[int],
                                bucket_batch_sizes: List[MonolingualBucketBatchSize],
                                target_vocab: vocab.Vocab,
                                max_seq_len_target: int,
                                batch_size: int,
                                fill_up: str) -> 'MonolingualSampleIter':
    """
    Returns a MonolingualSampleIter for the validation data.
    """
    logger.info("========================================")
    logger.info(" [LM] Creating validation data iterator ")
    logger.info("========================================")

    validation_target_sentences = data_io.SequenceReader(validation_target, target_vocab, add_bos=True, limit=None)

    validation_data_statistics = lm_get_data_statistics(validation_target_sentences,
                                                        buckets,
                                                        target_vocab)

    validation_data_statistics.log(bucket_batch_sizes)

    validation_data = data_loader.load(validation_target_sentences,
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    return MonolingualSampleIter(data=validation_data,
                                 buckets=buckets,
                                 batch_size=batch_size,
                                 bucket_batch_sizes=bucket_batch_sizes)


def define_monolingual_bucket_batch_sizes(buckets: List[int],
                                          batch_size: int,
                                          batch_by_words: bool,
                                          batch_num_devices: int,
                                          data_target_average_len: List[Optional[float]]) -> List[MonolingualBucketBatchSize]:
    """
    Computes bucket-specific batch sizes (sentences, average_words).

    If sentence-based batching: number of sentences is the same for each batch, determines the
    number of words. Hence all batch sizes for each bucket are equal.

    If word-based batching: number of sentences for each batch is set to the multiple of number
    of devices that produces the number of words closest to the target batch size.  Average
    target sentence length (non-padding symbols) is used for word number calculations.

    :param buckets: Bucket list.
    :param batch_size: Batch size.
    :param batch_by_words: Batch by words.
    :param batch_num_devices: Number of devices.
    :param data_target_average_len: Optional average target length for each bucket.
    """
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[MonolingualBucketBatchSize]
    largest_total_num_words = 0
    for buck_idx, bucket in enumerate(buckets):
        # Target/label length with padding
        padded_seq_len = bucket
        # Average target/label length excluding padding
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len
        average_seq_len = data_target_average_len[buck_idx]

        # Word-based: num words determines num sentences
        # Sentence-based: num sentences determines num words
        if batch_by_words:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            # Multiple of number of devices (int) closest to target number of words, assuming each sentence is of
            # average length
            batch_size_seq = batch_num_devices * round((batch_size / average_seq_len) / batch_num_devices)
            batch_size_word = batch_size_seq * average_seq_len
        else:
            batch_size_seq = batch_size
            batch_size_word = batch_size_seq * average_seq_len
        bucket_batch_sizes.append(MonolingualBucketBatchSize(bucket, batch_size_seq, batch_size_word))
        # Track largest number of word samples in a batch
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * bucket)

    # Final step: guarantee that largest bucket by sequence length also has largest total batch size.
    # When batching by sentences, this will already be the case.
    if batch_by_words:
        padded_seq_len = max(buckets[-1])
        average_seq_len = data_target_average_len[-1]
        while bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_num_words:
            bucket_batch_sizes[-1] = MonolingualBucketBatchSize(
                bucket_batch_sizes[-1].bucket,
                bucket_batch_sizes[-1].batch_size + batch_num_devices,
                bucket_batch_sizes[-1].average_words_per_batch + batch_num_devices * average_seq_len)
    return bucket_batch_sizes


def lm_get_training_data_iters(target: str,
                               validation_target: str,
                               target_vocab: vocab.Vocab,
                               target_vocab_path: Optional[str],
                               batch_size: int,
                               batch_by_words: bool,
                               batch_num_devices: int,
                               fill_up: str,
                               max_seq_len_target: int,
                               bucketing: bool,
                               bucket_width: int) -> Tuple['BaseMonolingualSampleIter',
                                                           'BaseMonolingualSampleIter',
                                                           'LMDataConfig', 'LMDataInfo']:
    """
    Returns data iterators for training and validation data.

    :param target: Path to training data.
    :param validation_target: Path to validation data.
    :param target_vocab: Vocabulary.
    :param target_vocab_path: Path to vocabulary.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len_target: Maximum sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: Tuple of (training data iterator, validation data iterator, data config, data info).
    """
    logger.info("======================================")
    logger.info(" [LM] Creating training data iterator ")
    logger.info("======================================")

    # Define buckets
    buckets = data_io.define_buckets(max_seq_len_target, bucket_width) if bucketing else [max_seq_len_target]

    # Input starts from <s>
    target_sentences = data_io.SequenceReader(target, target_vocab, add_bos=True)

    # Get data statistics
    data_statistics = lm_get_data_statistics(target_sentences, buckets, target_vocab)

    bucket_batch_sizes = define_monolingual_bucket_batch_sizes(buckets,
                                                               batch_size,
                                                               batch_by_words,
                                                               batch_num_devices,
                                                               data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    # </s> is added here in the output side (labels)
    data_loader = RawMonolingualDatasetLoader(buckets=buckets,
                                              eos_id=target_vocab[C.EOS_SYMBOL],
                                              pad_id=C.PAD_ID)

    training_data = data_loader.load(target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    data_info = LMDataInfo(target=target,
                           target_vocab=target_vocab_path)

    config_data = LMDataConfig(data_statistics=data_statistics,
                               max_seq_len_target=max_seq_len_target)

    train_iter = MonolingualSampleIter(data=training_data,
                                       buckets=buckets,
                                       batch_size=batch_size,
                                       bucket_batch_sizes=bucket_batch_sizes)

    validation_iter = lm_get_validation_data_iter(data_loader=data_loader,
                                                  validation_target=validation_target,
                                                  buckets=buckets,
                                                  bucket_batch_sizes=bucket_batch_sizes,
                                                  target_vocab=target_vocab,
                                                  max_seq_len_target=max_seq_len_target,
                                                  batch_size=batch_size,
                                                  fill_up=fill_up)

    return train_iter, validation_iter, config_data, data_info
