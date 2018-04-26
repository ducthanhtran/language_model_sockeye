import argparse
import os
from typing import cast, List, Tuple


from . import lm_common
from . import lm_data_io
from . import lm_model.TrainingLanguageModel
from sockeye import config
from sockeye.constants import BATCH_TYPE_WORD
from sockeye.vocab import vocab_from_json, load_or_create_vocab
from sockeye.utils import check_condition
from sockeye.training import EarlyStoppingTrainer

# from sockeye.arguments
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


# from sockeye.train
def lm_create_data_iters_and_vocabs(args: argparse.Namespace,
                                 resume_training: bool,
                                 output_folder: str) -> Tuple['data_io.BaseParallelSampleIter',
                                                              'data_io.BaseParallelSampleIter',
                                                              'data_io.DataConfig',
                                                              vocab.Vocab]:
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


# from sockeye.train
def lm_create_model_config(args: argparse.Namespace,
                        data_vocab_size: int,
                        config_data: data_io.DataConfig) -> model.ModelConfig:
    # TODO: has yet to be modified/checked upon
    num_embed = args.num_embed
    embed_dropout_source, embed_dropout_target = args.embed_dropout
    source_vocab_size, *source_factor_vocab_sizes = source_vocab_sizes

    check_encoder_decoder_args(args)

    config_conv = None

    config_encoder, encoder_num_hidden = create_encoder_config(args, config_conv)
    config_decoder = create_decoder_config(args, encoder_num_hidden)

    source_factor_configs = None
    if len(source_vocab_sizes) > 1:
        source_factor_configs = [encoder.FactorConfig(size, dim) for size, dim in zip(source_factor_vocab_sizes,
                                                                                      args.source_factors_num_embed)]

    config_embed_source = encoder.EmbeddingConfig(vocab_size=source_vocab_size,
                                                  num_embed=num_embed_source,
                                                  dropout=embed_dropout_source,
                                                  factor_configs=source_factor_configs)

    config_embed_target = encoder.EmbeddingConfig(vocab_size=target_vocab_size,
                                                  num_embed=num_embed_target,
                                                  dropout=embed_dropout_target)

    config_loss = loss.LossConfig(name=args.loss,
                                  vocab_size=target_vocab_size,
                                  normalization_type=args.loss_normalization_type,
                                  label_smoothing=args.label_smoothing)

    model_config = model.ModelConfig(config_data=config_data,
                                     vocab_source_size=source_vocab_size,
                                     vocab_target_size=target_vocab_size,
                                     config_embed_source=config_embed_source,
                                     config_embed_target=config_embed_target,
                                     config_encoder=config_encoder,
                                     config_decoder=config_decoder,
                                     config_loss=config_loss,
                                     weight_tying=args.weight_tying,
                                     weight_tying_type=args.weight_tying_type if args.weight_tying else None,
                                     weight_normalization=args.weight_normalization)
    return model_config


def lm_create_training_model():
    # TODO: implementation
    # TODO: check consistency with TrainingModel class (sockeye)
    pass


if __name__ == '__main__':
    args = create_parser().parse_args()

    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        train_iter, eval_iter, config_data, data_vocabs =
            lm_create_data_iters_and_vocabs(args=args,
                                            resume_training=resume_training,
                                            output_folder=output_folder)

        data_vocab_size = len(data_vocabs)
        logger.info('[LM] Vocabulary size: %s', data_vocab_size)

        lm_model_config = lm_create_model_config(args, data_vocab_size, config_data)
        lm_model_config.freeze()

        lm_training_model = lm_create_training_model(config=lm_model_config,
                                 context=context,
                                 output_dir=output_folder,
                                 train_iter=train_iter,
                                 args=args)

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
                       lr_decay_opt_states_reset=args.learning_rate_decay_param_reset,
                       decoder=create_checkpoint_decoder(args, exit_stack, context),
                       existing_parameters=args.params)
