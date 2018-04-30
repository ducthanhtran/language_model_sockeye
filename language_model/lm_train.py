import pdb # NOTE: debugging

import argparse
import logging
import os
import sys
from contextlib import ExitStack
from typing import cast, List, Tuple

import mxnet as mx

from . import lm_arguments
from . import lm_common
from . import lm_data_io
from . import lm_model

sys.path.append('../')

from sockeye import config
from sockeye.constants import BATCH_TYPE_WORD
from sockeye.data_io import BaseParallelSampleIter, DataConfig
from sockeye.encoder import EmbeddingConfig
from sockeye.loss import LossConfig
from sockeye.rnn import RNNConfig
from sockeye.vocab import Vocab, vocab_from_json, load_or_create_vocab
from sockeye.utils import check_condition
from sockeye.train import check_resume, determine_context, gradient_compression_params
from sockeye.training import EarlyStoppingTrainer

logger = logging.getLogger(__name__)


# from sockeye.train
def lm_create_data_iters_and_vocabs(args: argparse.Namespace,
                                    resume_training: bool,
                                    output_folder: str) -> Tuple['data_io.BaseParallelSampleIter',
                                                                 'data_io.BaseParallelSampleIter',
                                                                 'data_io.DataConfig',
                                                                 Vocab]:
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
    batch_by_words = args.batch_type == BATCH_TYPE_WORD

    # TODO: option arguments/constants should be well-structured
    train_data_error_msg = "Specify a LM training corpus with training and development data."
    check_condition(args.train_data is not None and args.dev_data is not None, train_data_error_msg)

    if resume_training:
        # Load the existing vocabs created when starting the training run.
        lm_vocab = vocab_from_json(os.path.join(output_folder, lm_common.LM_PREFIX + lm_common.VOCAB_NAME))

        # Recover the vocabulary path from the data info file:
        data_info = cast(lm_data_io.LanguageModelDataInfo,
                         Config.load(os.path.join(output_folder, lm_common.LM_DATA_INFO)))
        vocab_path = data_info.vocab

    else:
        # Load or create vocabs
        vocab_path = args.data_vocab
        vocab = load_or_create_vocab(
            data=args.train_data,
            vocab_path=vocab_path,
            num_words=num_words,
            word_min_count=word_min_count)

    train_iter, validation_iter, config_data, data_info = lm_data_io.lm_get_training_data_iters(
        train_data=os.path.abspath(args.train_data),
        validation_data=os.path.abspath(args.dev_data),
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
    logger.info("[LM] Writing LM data config to '%s'", data_info_fname)
    data_info.save(data_info_fname)

    return train_iter, validation_iter, config_data, vocab


# from sockeye.train
def lm_create_model_config(args: argparse.Namespace,
                           vocab_size: int,
                           config_data: DataConfig) -> lm_common.LanguageModelConfig:
    num_embed = args.num_embed
    embed_dropout = args.embed_dropout

    # TODO: adapt check_encoder_decoder_args(args) to LM
    # check_encoder_decoder_args(args)

    rnn_config = RNNConfig(cell_type=args.rnn_cell_type,
                           num_hidden=args.rnn_num_hidden,
                           num_layers=args.num_layers,
                           dropout_inputs=args.rnn_dropout_inputs,
                           dropout_states=args.rnn_dropout_states,
                           dropout_recurrent=args.rnn_dropout_recurrent,
                           residual=args.rnn_residual_connections,
                           first_residual_layer=args.rnn_first_residual_layer,
                           forget_bias=args.rnn_forget_bias)

    # TODO: weight tying

    config_embed = EmbeddingConfig(vocab_size=vocab_size,
                                   num_embed=num_embed,
                                   dropout=embed_dropout)

    config_loss = LossConfig(name=args.loss,
                             vocab_size=vocab_size,
                             normalization_type=args.loss_normalization_type,
                             label_smoothing=args.label_smoothing)

    model_config = lm_common.LanguageModelConfig(config_data=config_data,
                                                 vocab_size=vocab_size,
                                                 num_embed=num_embed,
                                                 rnn_config=rnn_config,
                                                 config_embed=config_embed,
                                                 config_loss=config_loss,
                                                 weight_tying=args.weight_tying)
    return model_config


def lm_create_training_model(config: lm_common.LanguageModelConfig,
                             context: List[mx.Context],
                             output_dir: str,
                             train_iter: BaseParallelSampleIter,
                             args: argparse.Namespace) -> lm_model.TrainingLanguageModel:
    """
    Create a training model and load the parameters from disk if needed.

    :param config: The configuration for the model.
    :param context: The context(s) to run on.
    :param output_dir: Output folder.
    :param train_iter: The training data iterator.
    :param args: Arguments as returned by argparse.
    :return: The training model.
    """
    training_model = lm_model.TrainingLanguageModel(config=config,
                                                    context=context,
                                                    output_dir=output_dir,
                                                    provide_data=train_iter.provide_data,
                                                    provide_label=train_iter.provide_label,
                                                    default_bucket_key=train_iter.default_bucket_key,
                                                    bucketing=not args.no_bucketing,
                                                    gradient_compression_params=gradient_compression_params(args),
                                                    fixed_param_names=[])

    return training_model


if __name__ == '__main__':
    args = lm_arguments.create_parser().parse_args()

    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        train_iter, eval_iter, config_data, data_vocab = lm_create_data_iters_and_vocabs(
            args=args,
            resume_training=resume_training,
            output_folder=output_folder)

        data_vocab_size = len(data_vocab)
        logger.info('[LM] Vocabulary size: %s', data_vocab_size)

        lm_model_config = lm_create_model_config(args, data_vocab_size, config_data)
        lm_model_config.freeze()

        lm_training_model = lm_create_training_model(config=lm_model_config,
                                                     context=context,
                                                     output_dir=output_folder,
                                                     train_iter=train_iter,
                                                     args=args)

        pdb.set_trace() # NOTE: BREAK HERE

        lm_trainer = sockeye.training.EarlyStoppingTrainer(model=lm_training_model,
                                                           optimizer_config=create_optimizer_config(args,ddata_vocab_size),
                                                           max_params_files_to_keep=args.keep_last_params)
        lm_trainer.fit(train_iter=train_iter,
                       validation_iter=eval_iter,
                       early_stopping_metric=args.optimized_metric,
                       metrics=args.metrics,
                       checkpoint_frequency=args.checkpoint_frequency,
                       max_num_not_improved=max_num_checkpoint_not_improved,
                       lr_decay_param_reset=args.learning_rate_decay_param_reset,
                       lr_decay_opt_states_reset=args.learning_rate_decay_optimizer_states_reset,
                       decoder=create_checkpoint_decoder(args, exit_stack, context),
                       existing_parameters=args.params)
