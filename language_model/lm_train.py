import pdb # NOTE: debugging

import argparse
import json
import logging
import os
import pickle
import random
import shutil
import sys
import time
from contextlib import ExitStack
from typing import cast, List, Optional, Tuple

from . import lm_arguments
from . import lm_common
from . import lm_data_io
from .lm_model import TrainingLanguageModel
from .lm_data_io import BaseMonolingualSampleIter, LMDataConfig, LMDataInfo, lm_get_training_data_iters

import mxnet as mx
import numpy as np

sys.path.append('../')

from sockeye import constants as C
from sockeye import config
from sockeye import loss
from sockeye import lr_scheduler
from sockeye import utils
from sockeye.encoder import EmbeddingConfig
from sockeye.optimizers import BatchState, CheckpointState, SockeyeOptimizer, OptimizerConfig
from sockeye.rnn import RNNConfig
from sockeye.vocab import Vocab, vocab_from_json, vocab_to_json, load_or_create_vocab
from sockeye.utils import check_condition
from sockeye.train import check_resume, create_optimizer_config, determine_context, gradient_compression_params
from sockeye.training import TrainState, Speedometer


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
    batch_by_words = args.batch_type == C.BATCH_TYPE_WORD

    train_data_error_msg = "Specify a LM training corpus with training and development/validation data."
    check_condition(args.train_data is not None and args.dev_data is not None, train_data_error_msg)

    if resume_training:
        # Load the existing vocabs created when starting the training run.
        target_vocab = vocab_from_json(os.path.join(output_folder, lm_common.LM_PREFIX + lm_common.LM_VOCAB_NAME))

        # Recover the vocabulary path from the data info file:
        data_info = cast(LMDataInfo,
                         Config.load(os.path.join(output_folder, lm_common.LM_DATA_INFO)))
        target_vocab_path = data_info.vocab

    else:
        # Load or create vocabs
        target_vocab_path = args.vocab
        target_vocab = load_or_create_vocab(data=args.train_data,
                                            vocab_path=target_vocab_path,
                                            num_words=num_words,
                                            word_min_count=word_min_count)

    train_iter, validation_iter, config_data, data_info = lm_get_training_data_iters(
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

    config_loss = loss.LossConfig(name=args.loss,
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
                             args: argparse.Namespace) -> TrainingLanguageModel:
    """
    Create a training model and load the parameters from disk if needed.

    :param config: The configuration for the model.
    :param context: The context(s) to run on.
    :param output_dir: Output folder.
    :param train_iter: The training data iterator.
    :param args: Arguments as returned by argparse.
    :return: The training model.
    """
    training_model = TrainingLanguageModel(config=config,
                                           context=context,
                                           output_dir=output_dir,
                                           provide_data=train_iter.provide_data,
                                           provide_label=train_iter.provide_label,
                                           default_bucket_key=train_iter.default_bucket_key,
                                           bucketing=not args.no_bucketing,
                                           gradient_compression_params=gradient_compression_params(args))
    # TODO: fix params

    return training_model


class LMEarlyStoppingTrainer:
    """
    Trainer class that fits a TrainingLanguageModel using early stopping on held-out validation data.

    :param model: TrainingLanguageModel instance.
    :param optimizer_config: The optimizer configuration.
    :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
    :param log_to_tensorboard: If True write training and evaluation logs to tensorboard event files.
    """

    def __init__(self,
                 model: TrainingLanguageModel,
                 optimizer_config: OptimizerConfig,
                 max_params_files_to_keep: int,
                 log_to_tensorboard: bool = True) -> None:
        self.model = model
        self.optimizer_config = optimizer_config
        self.max_params_files_to_keep = max_params_files_to_keep
        self.tensorboard_logger = None  # type: Optional[TensorboardLogger]
        # TODO: tensorboard
        self.state = None  # type: Optional[TrainState]

    def fit(self,
            train_iter: BaseMonolingualSampleIter,
            validation_iter: BaseMonolingualSampleIter,
            early_stopping_metric,
            metrics: List[str],
            checkpoint_frequency: int,
            max_num_not_improved: int,
            max_updates: Optional[int] = None,
            min_num_epochs: Optional[int] = None,
            max_num_epochs: Optional[int] = None,
            lr_decay_param_reset: bool = False,
            lr_decay_opt_states_reset: str = C.LR_DECAY_OPT_STATES_RESET_OFF,
            mxmonitor_pattern: Optional[str] = None,
            mxmonitor_stat_func: Optional[str] = None,
            allow_missing_parameters: bool = False,
            existing_parameters: Optional[str] = None):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder.

        :param train_iter: The training data iterator.
        :param validation_iter: The data iterator for held-out data.

        :param early_stopping_metric: The metric that is evaluated on held-out data and optimized.
        :param metrics: List of metrics that will be tracked during training.
        :param checkpoint_frequency: Frequency of checkpoints in number of update steps.

        :param max_num_not_improved: Stop training if early_stopping_metric did not improve for this many checkpoints.
               Use -1 to disable stopping based on early_stopping_metric.
        :param max_updates: Optional maximum number of update steps.
        :param min_num_epochs: Optional minimum number of epochs to train, overrides early stopping.
        :param max_num_epochs: Optional maximum number of epochs to train, overrides early stopping.

        :param lr_decay_param_reset: Reset parameters to previous best after a learning rate decay.
        :param lr_decay_opt_states_reset: How to reset optimizer states after a learning rate decay.

        :param mxmonitor_pattern: Optional pattern to match to monitor weights/gradients/outputs
               with MXNet's monitor. Default is None which means no monitoring.
        :param mxmonitor_stat_func: Choice of statistics function to run on monitored weights/gradients/outputs
               when using MXNEt's monitor.

        :param allow_missing_parameters: Allow missing parameters when initializing model parameters from file.
        :param existing_parameters: Optional filename of existing/pre-trained parameters to initialize from.

        :return: Best score on validation data observed during training.
        """
        self._check_args(metrics, early_stopping_metric, lr_decay_opt_states_reset, lr_decay_param_reset)
        logger.info("Early stopping by optimizing '%s'", early_stopping_metric)

        self._initialize_parameters(existing_parameters, allow_missing_parameters)
        self._initialize_optimizer()

        resume_training = os.path.exists(self.training_state_dirname)
        if resume_training:
            logger.info("Found partial training in '%s'. Resuming from saved state.", self.training_state_dirname)
            utils.check_condition('dist' not in self.optimizer_config.kvstore,
                                  "Training continuation not supported with distributed training.")
            self._load_training_state(train_iter)
        else:
            self.state = TrainState(early_stopping_metric)
            self._save_params()
            self._update_best_params_link()
            self._save_training_state(train_iter)
            self._update_best_optimizer_states(lr_decay_opt_states_reset)
            logger.info("Training started.")

        metric_train, metric_val, metric_loss = self._create_metrics(metrics, self.model.optimizer, self.model.loss)

        if mxmonitor_pattern is not None:
            self.model.install_monitor(mxmonitor_pattern, mxmonitor_stat_func)

        speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        tic = time.time()

        next_data_batch = train_iter.next()
        while True:

            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()
                if max_num_epochs is not None and self.state.epoch == max_num_epochs:
                    logger.info("Maximum # of epochs (%s) reached.", max_num_epochs)
                    break

            if max_updates is not None and self.state.updates == max_updates:
                logger.info("Maximum # of updates (%s) reached.", max_updates)
                break

            ######
            # STEP
            ######
            batch = next_data_batch
            self._step(self.model, batch, checkpoint_frequency, metric_train, metric_loss)
            if train_iter.iter_next():
                next_data_batch = train_iter.next()
                self.model.prepare_batch(next_data_batch)
            batch_num_samples = batch.data[0].shape[0]
            batch_num_tokens = batch.data[0].shape[1] * batch_num_samples
            self.state.updates += 1
            self.state.samples += batch_num_samples
            speedometer(self.state.epoch, self.state.updates, batch_num_samples, batch_num_tokens, metric_train)

            ############
            # CHECKPOINT
            ############
            if self.state.updates > 0 and self.state.updates % checkpoint_frequency == 0:
                time_cost = time.time() - tic
                self.state.checkpoint += 1
                # (1) save parameters and evaluate on validation data
                self._save_params()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f",
                            self.state.checkpoint, self.state.updates, self.state.epoch,
                            self.state.samples, time_cost, checkpoint_frequency / time_cost)
                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', self.state.checkpoint, name, val)
                self._evaluate(validation_iter, metric_val)
                for name, val in metric_val.get_name_value():
                    logger.info('Checkpoint [%d]\tValidation-%s=%f', self.state.checkpoint, name, val)

                # (3) update training metrics
                self._update_metrics(metric_train, metric_val)
                metric_train.reset()

                # (4) determine improvement
                has_improved = False
                previous_best = self.state.best_metric
                for checkpoint, metric_dict in enumerate(self.state.metrics, 1):
                    value = metric_dict.get("%s-val" % early_stopping_metric, self.state.best_metric)
                    if utils.metric_value_is_better(value, self.state.best_metric, early_stopping_metric):
                        self.state.best_metric = value
                        self.state.best_checkpoint = checkpoint
                        has_improved = True

                if has_improved:
                    self._update_best_params_link()
                    self._update_best_optimizer_states(lr_decay_opt_states_reset)
                    self.state.num_not_improved = 0
                    logger.info("Validation-%s improved to %f (delta=%f).", early_stopping_metric,
                                self.state.best_metric, abs(self.state.best_metric - previous_best))
                else:
                    self.state.num_not_improved += 1
                    logger.info("Validation-%s has not improved for %d checkpoints, best so far: %f",
                                early_stopping_metric, self.state.num_not_improved, self.state.best_metric)

                # If using an extended optimizer, provide extra state information about the current checkpoint
                # Loss: optimized metric
                if metric_loss is not None and isinstance(self.model.optimizer, SockeyeOptimizer):
                    m_val = 0
                    for name, val in metric_val.get_name_value():
                        if name == early_stopping_metric:
                            m_val = val
                    checkpoint_state = CheckpointState(checkpoint=self.state.checkpoint, metric_val=m_val)
                    self.model.optimizer.pre_update_checkpoint(checkpoint_state)

                # (5) adjust learning rates
                self._adjust_learning_rate(has_improved, lr_decay_param_reset, lr_decay_opt_states_reset)

                # (6) save training state
                self._save_training_state(train_iter)

                # (7) determine stopping
                if 0 <= max_num_not_improved <= self.state.num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, self.state.num_not_improved)
                    stop_fit = True

                    if min_num_epochs is not None and self.state.epoch < min_num_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_num_epochs, self.state.epoch)
                        stop_fit = False

                    if stop_fit:
                        break

                tic = time.time()

        self._cleanup(lr_decay_opt_states_reset)
        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.state.best_checkpoint, early_stopping_metric, self.state.best_metric)
        return self.state.best_metric

    def _step(self,
              model: TrainingLanguageModel,
              batch: mx.io.DataBatch,
              checkpoint_frequency: int,
              metric_train: mx.metric.EvalMetric,
              metric_loss: Optional[mx.metric.EvalMetric] = None):
        """
        Performs an update to model given a batch and updates metrics.
        """

        if model.monitor is not None:
            model.monitor.tic()

        ####################
        # Forward & Backward
        ####################
        model.run_forward_backward(batch, metric_train)

        ####################
        # Gradient rescaling
        ####################
        gradient_norm = None
        if self.state.updates > 0 and (self.state.updates + 1) % checkpoint_frequency == 0:
            # compute values for logging to metrics (before rescaling...)
            gradient_norm = self.state.gradient_norm = model.get_global_gradient_norm()

        # note: C.GRADIENT_CLIPPING_TYPE_ABS is handled by the mxnet optimizer directly
        if self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_NORM:
            if gradient_norm is None:
                gradient_norm = model.get_global_gradient_norm()
            # clip gradients
            if gradient_norm > self.optimizer_config.gradient_clipping_threshold:
                ratio = self.optimizer_config.gradient_clipping_threshold / gradient_norm
                model.rescale_gradients(ratio)

        # If using an extended optimizer, provide extra state information about the current batch
        optimizer = model.optimizer
        if metric_loss is not None and isinstance(optimizer, SockeyeOptimizer):
            # Loss for this batch
            metric_loss.reset()
            metric_loss.update(batch.label, model.module.get_outputs())
            [(_, m_val)] = metric_loss.get_name_value()
            batch_state = BatchState(metric_val=m_val)
            optimizer.pre_update_batch(batch_state)

        ########
        # UPDATE
        ########
        model.update()

        if model.monitor is not None:
            results = model.monitor.toc()
            if results:
                for _, k, v in results:
                    logger.info('Monitor: Batch [{:d}] {:s} {:s}'.format(self.state.updates, k, v))

    def _evaluate(self, val_iter: BaseMonolingualSampleIter, val_metric: mx.metric.EvalMetric):
        """
        Evaluates the model on the validation data and updates the validation metric(s).
        """
        val_iter.reset()
        val_metric.reset()
        self.model.evaluate(val_iter, val_metric)

    def _update_metrics(self,
                        metric_train: mx.metric.EvalMetric,
                        metric_val: mx.metric.EvalMetric):
        """
        Updates metrics for current checkpoint.
        Writes all metrics to the metrics file and optionally logs to tensorboard.
        """
        checkpoint_metrics = {"epoch": self.state.epoch,
                              "learning-rate": self.model.optimizer.learning_rate,
                              "gradient-norm": self.state.gradient_norm,
                              "time-elapsed": time.time() - self.state.start_tic}
        gpu_memory_usage = utils.get_gpu_memory_usage(self.model.context)
        if gpu_memory_usage is not None:
            checkpoint_metrics['used-gpu-memory'] = sum(v[0] for v in gpu_memory_usage.values())

        for name, value in metric_train.get_name_value():
            checkpoint_metrics["%s-train" % name] = value
        for name, value in metric_val.get_name_value():
            checkpoint_metrics["%s-val" % name] = value

        self.state.metrics.append(checkpoint_metrics)
        utils.write_metrics_file(self.state.metrics, self.metrics_fname)
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_metrics(checkpoint_metrics, self.state.checkpoint)

    def _cleanup(self, lr_decay_opt_states_reset: str):
        """
        Cleans parameter files, training state directory and waits for remaining decoding processes.
        """
        utils.cleanup_params_files(self.model.output_dir, self.max_params_files_to_keep,
                                   self.state.checkpoint, self.state.best_checkpoint)

        final_training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(final_training_state_dirname):
            shutil.rmtree(final_training_state_dirname)
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            best_opt_states_fname = os.path.join(self.model.output_dir, C.OPT_STATES_BEST)
            if os.path.exists(best_opt_states_fname):
                os.remove(best_opt_states_fname)

    def _initialize_parameters(self, params: Optional[str], allow_missing_params: bool):
        self.model.initialize_parameters(self.optimizer_config.initializer, allow_missing_params)
        if params is not None:
            logger.info("Training will start with parameters loaded from '%s'", params)
            self.model.load_params_from_file(params, allow_missing_params=allow_missing_params)
        self.model.log_parameters()

    def _initialize_optimizer(self):
        self.model.initialize_optimizer(self.optimizer_config)

    def _adjust_learning_rate(self, has_improved: bool, lr_decay_param_reset: bool, lr_decay_opt_states_reset: str):
        """
        Adjusts the optimizer learning rate if required.
        """
        if self.optimizer_config.lr_scheduler is not None:
            if issubclass(type(self.optimizer_config.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler):
                lr_adjusted = self.optimizer_config.lr_scheduler.new_evaluation_result(has_improved)  # type: ignore
            else:
                lr_adjusted = False
            if lr_adjusted and not has_improved:
                if lr_decay_param_reset:
                    logger.info("Loading parameters from last best checkpoint: %d",
                                self.state.best_checkpoint)
                    self.model.load_params_from_file(self.best_params_fname)
                if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
                    logger.info("Loading initial optimizer states")
                    self.model.load_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_INITIAL))
                elif lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
                    logger.info("Loading optimizer states from best checkpoint: %d",
                                self.state.best_checkpoint)
                    self.model.load_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_BEST))

    @property
    def best_params_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.PARAMS_BEST_NAME)

    @property
    def current_params_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.PARAMS_NAME % self.state.checkpoint)

    @property
    def metrics_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.METRICS_NAME)

    @property
    def training_state_dirname(self) -> str:
        return os.path.join(self.model.output_dir, C.TRAINING_STATE_DIRNAME)

    @staticmethod
    def _create_eval_metric(metric_name: str) -> mx.metric.EvalMetric:
        """
        Creates an EvalMetric given a metric names.
        """
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        if metric_name == C.ACCURACY:
            return utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])
        elif metric_name == C.PERPLEXITY:
            return mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])
        else:
            raise ValueError("unknown metric name")

    @staticmethod
    def _create_eval_metric_composite(metric_names: List[str]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = [LMEarlyStoppingTrainer._create_eval_metric(metric_name) for metric_name in metric_names]
        return mx.metric.create(metrics)

    def _create_metrics(self, metrics: List[str], optimizer: mx.optimizer.Optimizer,
                        loss: loss.Loss) -> Tuple[mx.metric.EvalMetric,
                                                  mx.metric.EvalMetric,
                                                  Optional[mx.metric.EvalMetric]]:
        metric_train = self._create_eval_metric_composite(metrics)
        metric_val = self._create_eval_metric_composite(metrics)
        # If optimizer requires it, track loss as metric
        if isinstance(optimizer, SockeyeOptimizer):
            if optimizer.request_optimized_metric:
                metric_loss = self._create_eval_metric(self.state.early_stopping_metric)
            else:
                metric_loss = loss.create_metric()
        else:
            metric_loss = None
        return metric_train, metric_val, metric_loss

    def _update_best_params_link(self):
        """
        Updates the params.best link to the latest best parameter file.
        """
        best_params_path = self.best_params_fname
        actual_best_params_fname = C.PARAMS_NAME % self.state.best_checkpoint
        if os.path.lexists(best_params_path):
            os.remove(best_params_path)
        os.symlink(actual_best_params_fname, best_params_path)

    def _update_best_optimizer_states(self, lr_decay_opt_states_reset: str):
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            self.model.save_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_BEST))

    def _save_initial_optimizer_states(self, lr_decay_opt_states_reset: str):
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
            self.model.save_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_INITIAL))

    def _check_args(self,
                    metrics: List[str],
                    early_stopping_metric: str,
                    lr_decay_opt_states_reset: str,
                    lr_decay_param_reset: bool):
        """
        Helper function that checks various configuration compatibilities.
        """
        utils.check_condition(early_stopping_metric in metrics, "Early stopping metric must be tracked.")
        utils.check_condition(len(metrics) > 0, "At least one metric must be provided.")
        for metric in metrics:
            utils.check_condition(metric in C.METRICS, "Unknown metric to track during training: %s" % metric)

        if 'dist' in self.optimizer_config.kvstore:
            # In distributed training the optimizer will run remotely. For eve we however need to pass information about
            # the loss, which is not possible anymore by means of accessing self.module._curr_module._optimizer.
            utils.check_condition(self.optimizer_config.name != C.OPTIMIZER_EVE,
                                  "Eve optimizer not supported with distributed training.")
            utils.check_condition(
                not issubclass(type(self.optimizer_config.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler),
                "Adaptive learning rate schedulers not supported with a dist kvstore. "
                "Try a fixed schedule such as %s." % C.LR_SCHEDULER_FIXED_RATE_INV_SQRT_T)
            utils.check_condition(not lr_decay_param_reset, "Parameter reset when the learning rate decays not "
                                                            "supported with distributed training.")
            utils.check_condition(lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_OFF,
                                  "Optimizer state reset when the learning rate decays "
                                  "not supported with distributed training.")

        utils.check_condition(self.optimizer_config.gradient_clipping_type in C.GRADIENT_CLIPPING_TYPES,
                              "Unknown gradient clipping type %s" % self.optimizer_config.gradient_clipping_type)

        utils.check_condition(early_stopping_metric in C.METRICS,
                              "Unsupported early-stopping metric: %s" % early_stopping_metric)

    def _save_params(self):
        """
        Saves model parameters at current checkpoint and optionally cleans up older parameter files to save disk space.
        """
        self.model.save_params_to_file(self.current_params_fname)
        utils.cleanup_params_files(self.model.output_dir, self.max_params_files_to_keep, self.state.checkpoint,
                                   self.state.best_checkpoint)

    def _save_training_state(self, train_iter: BaseMonolingualSampleIter):
        """
        Saves current training state.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)

        # (1) Parameters: link current file
        params_base_fname = C.PARAMS_NAME % self.state.checkpoint
        os.symlink(os.path.join("..", params_base_fname),
                   os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME))

        # (2) Optimizer states
        opt_state_fname = os.path.join(training_state_dirname, C.OPT_STATES_LAST)
        self.model.save_optimizer_states(opt_state_fname)

        # (3) Data iterator
        train_iter.save_state(os.path.join(training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # (4) Random generators
        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp)

        # (5) Training state
        self.state.save(os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # (6) Learning rate scheduler
        with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
            pickle.dump(self.optimizer_config.lr_scheduler, fp)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(self.training_state_dirname):
            os.rename(self.training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, self.training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            shutil.rmtree(delete_training_state_dirname)

    def _load_training_state(self, train_iter: BaseMonolingualSampleIter):
        """
        Loads the full training state from disk.

        :param train_iter: training data iterator.
        """
        # (1) Parameters
        params_fname = os.path.join(self.training_state_dirname, C.TRAINING_STATE_PARAMS_NAME)
        self.model.load_params_from_file(params_fname)

        # (2) Optimizer states
        opt_state_fname = os.path.join(self.training_state_dirname, C.OPT_STATES_LAST)
        self.model.load_optimizer_states(opt_state_fname)

        # (3) Data Iterator
        train_iter.load_state(os.path.join(self.training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # (4) Random generators
        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(self.training_state_dirname, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            np.random.set_state(pickle.load(fp))

        # (5) Training state
        self.state = TrainState.load(os.path.join(self.training_state_dirname, C.TRAINING_STATE_NAME))

        # (6) Learning rate scheduler
        with open(os.path.join(self.training_state_dirname, C.SCHEDULER_STATE_NAME), "rb") as fp:
            self.optimizer_config.lr_scheduler = pickle.load(fp)
        # initialize optimizer again
        self.model.initialize_optimizer(self.optimizer_config)


if __name__ == '__main__':
    args = lm_arguments.create_parser().parse_args()

    utils.seedRNGs(args.seed)

    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    utils.log_basic_info(args)
    with open(os.path.join(output_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        train_iter, eval_iter, config_data, target_vocab = lm_create_data_iters_and_vocabs(
            args=args,
            resume_training=resume_training,
            output_folder=output_folder)

        # Dump the vocabularies if we're just starting up
        if not resume_training:
            vocab_to_json(target_vocab, os.path.join(output_folder, lm_common.LM_PREFIX + lm_common.LM_VOCAB_NAME))

        target_vocab_size = len(target_vocab)
        logger.info('[LM] Vocabulary size: %s', target_vocab_size)

        lm_model_config = lm_create_model_config(args, target_vocab_size, config_data)
        lm_model_config.freeze()

        lm_training_model = lm_create_training_model(config=lm_model_config,
                                                     context=context,
                                                     output_dir=output_folder,
                                                     train_iter=train_iter,
                                                     args=args)

        # Handle options that override training settings
        max_updates = args.max_updates
        max_num_checkpoint_not_improved = args.max_num_checkpoint_not_improved
        min_num_epochs = args.min_num_epochs
        max_num_epochs = args.max_num_epochs
        if min_num_epochs is not None and max_num_epochs is not None:
            check_condition(min_num_epochs <= max_num_epochs,
                            "Minimum number of epochs must be smaller than maximum number of epochs")
        # Fixed training schedule always runs for a set number of updates
        if args.learning_rate_schedule:
            max_updates = sum(num_updates for (_, num_updates) in args.learning_rate_schedule)
            max_num_checkpoint_not_improved = -1
            min_num_epochs = None
            max_num_epochs = None

        lm_trainer = LMEarlyStoppingTrainer(model=lm_training_model,
                                            optimizer_config=create_optimizer_config(args, [target_vocab_size]),
                                            max_params_files_to_keep=args.keep_last_params,
                                            log_to_tensorboard=False)

        lm_trainer.fit(train_iter=train_iter,
                       validation_iter=eval_iter,
                       early_stopping_metric=args.optimized_metric,
                       metrics=args.metrics,
                       checkpoint_frequency=args.checkpoint_frequency,
                       max_num_not_improved=max_num_checkpoint_not_improved,
                       lr_decay_param_reset=args.learning_rate_decay_param_reset,
                       lr_decay_opt_states_reset=args.learning_rate_decay_optimizer_states_reset,
                       existing_parameters=args.params)
