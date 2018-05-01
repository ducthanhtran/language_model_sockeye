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
from .lm_data_io import BaseMonolingualSampleIter, LMDataConfig

sys.path.append('../')

from sockeye import config
from sockeye.constants import BATCH_TYPE_WORD
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
                                    output_folder: str) -> Tuple['BaseMonolingualSampleIter',
                                                                 'BaseMonolingualSampleIter',
                                                                 'LMDataConfig',
                                                                 Vocab]:
    """
    Create the data iterators and the vocabulary.

    :param args: Arguments as returned by argparse.
    :param resume_training: Whether to resume training.
    :param output_folder: Output folder.
    :return: The data iterators (train, validation, config_data) as well as the vocabulary.
    """
    max_seq_len_target = args.max_seq_len
    num_words = args.num_words
    word_min_count = args.word_min_count
    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    batch_by_words = args.batch_type == BATCH_TYPE_WORD

    # TODO: option arguments/constants should be well-structured
    train_data_error_msg = "Specify a LM training corpus with training and development data."
    check_condition(args.train_data is not None and args.dev_data is not None, train_data_error_msg)

    if resume_training:
        # Load the existing vocabs created when starting the training run.
        target_vocab = vocab_from_json(os.path.join(output_folder, lm_common.LM_PREFIX + lm_common.VOCAB_NAME))

        # Recover the vocabulary path from the data info file:
        data_info = cast(lm_data_io.LMDataInfo,
                         Config.load(os.path.join(output_folder, lm_common.LM_DATA_INFO)))
        target_vocab_path = data_info.vocab

    else:
        # Load or create vocabs
        target_vocab_path = args.vocab
        target_vocab = load_or_create_vocab(data=args.train_data,
                                            vocab_path=target_vocab_path,
                                            num_words=num_words,
                                            word_min_count=word_min_count)

    train_iter, validation_iter, config_data, data_info = lm_data_io.lm_get_training_data_iters(
        target=os.path.abspath(args.train_data),
        validation_target=os.path.abspath(args.dev_data),
        target_vocab=target_vocab,
        target_vocab_path=target_vocab_path,
        batch_size=args.batch_size,
        batch_by_words=batch_by_words,
        batch_num_devices=batch_num_devices,
        fill_up=args.fill_up,
        max_seq_len_target=max_seq_len_target,
        bucketing=not args.no_bucketing,
        bucket_width=args.bucket_width)

    data_info_fname = os.path.join(output_folder, lm_common.LM_DATA_INFO)
    logger.info("[LM] Writing LM data config to '%s'", data_info_fname)
    data_info.save(data_info_fname)

    return train_iter, validation_iter, config_data, target_vocab


# from sockeye.train
def lm_create_model_config(args: argparse.Namespace,
                           target_vocab_size: int,
                           config_data: LMDataConfig) -> lm_common.LMConfig:
    num_embed_target = args.num_embed
    embed_dropout_target = args.embed_dropout

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

    config_embed = EmbeddingConfig(vocab_size=target_vocab_size,
                                   num_embed=num_embed_target,
                                   dropout=embed_dropout_target)

    config_loss = LossConfig(name=args.loss,
                             vocab_size=target_vocab_size,
                             normalization_type=args.loss_normalization_type,
                             label_smoothing=args.label_smoothing)

    model_config = lm_common.LMConfig(config_data=config_data,
                                      target_vocab_size=target_vocab_size,
                                      num_embed_target=num_embed_target,
                                      rnn_config=rnn_config,
                                      config_embed=config_embed,
                                      config_loss=config_loss,
                                      weight_tying=args.weight_tying)
    return model_config


def lm_create_training_model(config: lm_common.LMConfig,
                             context: List[mx.Context],
                             output_dir: str,
                             train_iter: BaseMonolingualSampleIter,
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
                                                    gradient_compression_params=gradient_compression_params(args))
    # TODO: fix params

    return training_model


if __name__ == '__main__':
    args = lm_arguments.create_parser().parse_args()

    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        train_iter, eval_iter, config_data, target_vocab = lm_create_data_iters_and_vocabs(
            args=args,
            resume_training=resume_training,
            output_folder=output_folder)

        target_vocab_size = len(target_vocab)
        logger.info('[LM] Vocabulary size: %s', target_vocab_size)

        lm_model_config = lm_create_model_config(args, target_vocab_size, config_data)
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
