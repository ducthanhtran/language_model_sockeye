import argparse
import os
from typing import cast, List, Tuple


from . import lm_common
from . import lm_data_io
from . import lm_model.TrainingLanguageModel
from sockeye import config
from sockeye import constants as C
from sockeye.vocab import vocab_from_json, load_or_create_vocab
from sockeye.utils import check_condition

# from Sockeye.arguments
def regular_file() -> Callable:
    def check_regular_file(value_to_check):
        value_to_check = str(value_to_check)
        if not os.path.isfile(value_to_check):
            raise argparse.ArgumentTypeError("must exist and be a regular file.")
        return value_to_check
    return check_regular_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data',
                        required=True,
                        type=regular_file(),
                        help='training data. Target labels are generated')
    parser.add_argument('--dev-data',
                        required=True,
                        type=regular_file(),
                        help='development data - used for early stopping')
    return parser

def create_data_iters_and_vocabs(args: argparse.Namespace,
                                 resume_training: bool,
                                 output_folder: str) -> Tuple['data_io.BaseParallelSampleIter',
                                                              'data_io.BaseParallelSampleIter',
                                                              'data_io.DataConfig',
                                                              List[vocab.Vocab], vocab.Vocab]:
    """
    Create the data iterators and the vocabulary.

    :param args: Arguments as returned by argparse.
    :param resume_training: Whether to resume training.
    :param output_folder: Output folder.
    :return: The data iterators (train, validation, config_data) as well as the vocabulary.
    """
    max_seq_len = args.max_seq_len
    num_words = args.num_words
    word_min_count = args.word_min_count
    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    batch_by_words = args.batch_type == C.BATCH_TYPE_WORD


    # TODO: option arguments/constants should be well-structured
    train_data_error_msg = "Specify a LM training corpus with training and development data."
    check_condition(args.train_data is None and args.dev_data is None, train_data_error_msg)

    if resume_training:
        # Load the existing vocabs created when starting the training run.
        lm_vocab = vocab_from_json(os.path.join(output_folder, lm_common.LM_PREFIX + lm_common.VOCAB_NAME))

        # Recover the vocabulary path from the data info file:
        data_info = cast(lm_data_io.LanguageModelDataInfo,
                         Config.load(os.path.join(output_folder, lm_common.LM_DATA_INFO)))
        vocab_path = data_info.vocab

    else:
        # Load or create vocabs
        vocab_path = args.vocab
        vocab = load_or_create_vocab(
            data=args.train_data,
            vocab_path=vocab_path,
            num_words=num_words,
            word_min_count=word_min_count)

    # No factors for train/validation data
    train_iter, validation_iter, config_data, data_info = lm_data_io.lm_get_training_data_iters(
        train_data=os.path.abspath(args.train_data),
        validation_data=os.path.abspath(args.validation_data),
        vocab=vocab,
        vocab_path=vocab_path,
        batch_size=args.batch_size,
        batch_by_words=batch_by_words,
        batch_num_devices=batch_num_devices,
        fill_up=args.fill_up,
        max_seq_len=max_seq_len,
        bucketing=not args.no_bucketing,
        bucket_width=args.bucket_width)

    data_info_fname = os.path.join(output_folder, lm_common.LM_DATA_INFO)
    logger.info("Writing LM data config to '%s'", data_info_fname)
    data_info.save(data_info_fname)

    return train_iter, validation_iter, config_data, vocab


if __name__ == '__main__':
    args = create_parser().parse_args()
    # TODO: perform training on lm_model.TrainingLanguageModel
