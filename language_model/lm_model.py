from . import lm_common
from . import lm_decoder
from . import lm_model
from sockeye.encoder import Embedding
from sockeye.layers import OutputLayer
from sockeye import constants as C

import mxnet as mx


class LanguageModel:
    """
    LanguageModel shares components needed for both training and inference.
    The main components of a LanguageModel are
    1) Embedding
    2) Decoder
    3) Output layer

    LanguageModel conatins parameters and their values that are fixed at training time and must be re-used at inference time.

    :param config: Model configuration.
    """

    def __init__(self, config: lm_common.LanguageModelConfig) -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()
        logger.info("%s", self.config)

        # decoder first (to know the decoder depth)
        self.decoder = LanguageModelDecoder(config=self.config,
                                            prefix=LM_PREFIX + "decoder_")

        # embedding
        embed_weight, out_weight = self._get_embed_weights()
        self.embedding = Embedding(self.config.config_embed,
                                   prefix=LM_PREFIX + "embed_",
                                   embed_weight=embed_weight)

        # output layer
        self.output_layer = OutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                        vocab_size=self.config.vocab_size,
                                        weight=out_weight,
                                        weight_normalization=False)

        self.params = None  # type: Optional[Dict]
        self.aux_params = None  # type: Optional[Dict]

    def save_config(self, folder: str):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, LM_PREFIX + C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> lm_common.LanguageModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = lm_common.LanguageModelConfig.load(fname)
        logger.info('ModelConfig loaded from "%s"', fname)
        return cast(lm_common.LanguageModelConfig, config)  # type: ignore

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        if self.aux_params is not None:
            utils.save_params(self.params.copy(), fname, self.aux_params.copy())
        else:
            utils.save_params(self.params.copy(), fname)
        logging.info('Saved LM params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        utils.check_condition(os.path.exists(fname), "No LM parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.params, self.aux_params = utils.load_params(fname)
        logger.info('Loaded LM params from "%s"', fname)

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, LM_PREFIX + C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embed_weights(self) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Returns embedding parameters.

        :return: Tuple of parameter symbols.
        """
        w_embed = mx.sym.Variable(LM_PREFIX + "embed_weight",
                                  shape=(self.config.vocab_size, self.config.num_embed))
        w_out = mx.sym.Variable(LM_PREFIX + "output_weight",
                                shape=(self.config.vocab_size, self.decoder.get_num_hidden()))

        if self.config.weight_tying:
            logger.info("Tying the LM embeddings and output layer parameters.")
            utils.check_condition(self.config.num_embed == self.decoder.get_num_hidden(),
                                  "Weight tying requires LM embedding size and LM decoder hidden size " +
                                  "to be equal: %d vs. %d" % (self.config.num_embed,
                                                              self.decoder.get_num_hidden()))
            w_out = w_embed

        return w_embed, w_out

class TrainingLanguageModel(lm_model.LanguageModel):
    """
    TrainingLanguageModel is a LanguageModel that fully unrolls over input and output sequences.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU).
    :param output_dir: Directory where this model is stored.
    :param provide_data: List of input data descriptions.
    :param provide_label: List of label descriptions.
    :param default_bucket_key: Default bucket key.
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param gradient_compression_params: Optional dictionary of gradient compression parameters.
    :param fixed_param_names: Optional list of params to fix during training (i.e. their values will not be trained).
    """

    def __init__(self,
                 config: lm_common.LanguageModelConfig,
                 context: List[mx.context.Context],
                 output_dir: str,
                 provide_data: List[mx.io.DataDesc],
                 provide_label: List[mx.io.DataDesc],
                 default_bucket_key: int,
                 bucketing: bool,
                 gradient_compression_params: Optional[Dict[str, Any]] = None,
                 fixed_param_names: Optional[List[str]] = None) -> None:
        super().__init__(config)
        self.context = context
        self.output_dir = output_dir
        self.fixed_param_names = fixed_param_names
        self._bucketing = bucketing
        self._gradient_compression_params = gradient_compression_params
        self._initialize(provide_data, provide_label, default_bucket_key)
        self._monitor = None  # type: Optional[mx.monitor.Monitor]

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    provide_label: List[mx.io.DataDesc],
                    default_bucket_key: int):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        input = mx.sym.Variable(lm_common.LM_PREFIX + "input")
        input_words = input.split(num_outputs=self.config.config_embed.num_factors,
                                  axis=2, squeeze_axis=True)[0]
        input_length = utils.compute_lengths(input_words)
        output = mx.sym.Variable(lm_common.LM_PREFIX + "output")
        output_length = utils.compute_lengths(output)
        labels = mx.sym.reshape(data=mx.sym.Variable(lm_common.LM_PREFIX + "label"),
                                shape=(-1,))

        self.model_loss = loss.get_loss(self.config.config_loss)

        data_names = [lm_common.LM_PREFIX + "input", lm_common.LM_PREFIX + "output"]
        label_names = [lm_common.LM_PREFIX + "label"]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))
        provide_label_names = [d[0] for d in provide_label]
        utils.check_condition(provide_label_names == label_names,
                              "incompatible provide_label: %s, names should be %s" % (provide_label_names, label_names))

        def sym_gen(seq_len):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """

            # input embedding
            # input_embed: (batch_size, input_embed_length, num_embed)
            (input_embed,
             input_embed_length,
             input_embed_seq_len) = self.embedding.encode(input, input_length, seq_len)

            # output embedding
            # output_embed: (batch_size, output_embed_length, num_embed)
            (output_embed,
             output_embed_length,
             output_embed_seq_len) = self.embedding.encode(output, output_length, seq_len)

            # decoder
            # decoded: (batch-size, output_len, decoder_depth)
            decoded = self.decoder.decode_sequence(input_embed, input_embed_length, input_embed_seq_len,
                                                   output_embed, output_embed_length, output_embed_seq_len)

            # decoded: (batch_size * seq_len, decoder_depth)
            decoded = mx.sym.reshape(data=decoded, shape=(-3, 0))

            # output layer
            # logits: (batch_size * seq_len, vocab_size)
            logits = self.output_layer(decoded)

            probs = self.model_loss.get_loss(logits, labels)

            return mx.sym.Group(probs), data_names, label_names

        if self._bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", default_bucket_key)
            self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=default_bucket_key,
                                                 context=self.context,
                                                 compression_params=self._gradient_compression_params,
                                                 fixed_param_names=self.fixed_param_names)
        else:
            logger.info("No bucketing. Unrolled to (%d)",
                        self.config.max_seq_len)
            symbol, _, __ = sym_gen(default_bucket_key)
            self.module = mx.mod.Module(symbol=symbol,
                                        data_names=data_names,
                                        label_names=label_names,
                                        logger=logger,
                                        context=self.context,
                                        compression_params=self._gradient_compression_params,
                                        fixed_param_names=self.fixed_param_names)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=provide_label,
                         for_training=True,
                         force_rebind=True,
                         grad_req='write')

        self.module.symbol.save(os.path.join(self.output_dir, C.SYMBOL_NAME))

        self.save_version(self.output_dir)
        self.save_config(self.output_dir)

    def run_forward_backward(self, batch: mx.io.DataBatch, metric: mx.metric.EvalMetric):
        """
        Runs forward/backward pass and updates training metric(s).
        """
        self.module.forward_backward(batch)
        self.module.update_metric(metric, batch.label)

    def update(self):
        """
        Updates parameters of the module.
        """
        self.module.update()

    def get_global_gradient_norm(self) -> float:
        """
        Returns global gradient norm.
        """
        # average norm across executors:
        exec_norms = [global_norm([arr for arr in exe.grad_arrays if arr is not None]) for exe in self.executors]
        norm_val = sum(exec_norms) / float(len(exec_norms))
        norm_val *= self.optimizer.rescale_grad
        return norm_val

    def rescale_gradients(self, scale: float):
        """
        Rescales gradient arrays of executors by scale.
        """
        for exe in self.executors:
            for arr in exe.grad_arrays:
                if arr is None:
                    continue
                arr *= scale

    def prepare_batch(self, batch: mx.io.DataBatch):
        """
        Pre-fetches the next mini-batch.

        :param batch: The mini-batch to prepare.
        """
        self.module.prepare(batch)

    def evaluate(self, eval_iter: data_io.BaseParallelSampleIter, eval_metric: mx.metric.EvalMetric):
        """
        Resets and recomputes evaluation metric on given data iterator.
        """
        for eval_batch in eval_iter:
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(eval_metric, eval_batch.label)

    @property
    def current_module(self) -> mx.module.Module:
        # As the BucketingModule does not expose all methods of the underlying Module we need to directly access
        # the currently active module, when we use bucketing.
        return self.module._curr_module if self._bucketing else self.module

    @property
    def executors(self):
        return self.current_module._exec_group.execs

    @property
    def loss(self):
        return self.model_loss

    @property
    def optimizer(self) -> Union[mx.optimizer.Optimizer, SockeyeOptimizer]:
        """
        Returns the optimizer of the underlying module.
        """
        # TODO: Push update to MXNet to expose the optimizer (Module should have a get_optimizer method)
        return self.current_module._optimizer

    def initialize_optimizer(self, config: OptimizerConfig):
        """
        Initializes the optimizer of the underlying module with an optimizer config.
        """
        self.module.init_optimizer(kvstore=config.kvstore,
                                   optimizer=config.name,
                                   optimizer_params=config.params,
                                   force_init=True)  # force init for training resumption use case

    def save_optimizer_states(self, fname: str):
        """
        Saves optimizer states to a file.

        :param fname: File name to save optimizer states to.
        """
        self.current_module.save_optimizer_states(fname)

    def load_optimizer_states(self, fname: str):
        """
        Loads optimizer states from file.

        :param fname: File name to load optimizer states from.
        """
        self.current_module.load_optimizer_states(fname)

    def initialize_parameters(self, initializer: mx.init.Initializer, allow_missing_params: bool):
        """
        Initializes the parameters of the underlying module.

        :param initializer: Parameter initializer.
        :param allow_missing_params: Whether to allow missing parameters.
        """
        self.module.init_params(initializer=initializer,
                                arg_params=self.params,
                                aux_params=self.aux_params,
                                allow_missing=allow_missing_params,
                                force_init=False)

    def log_parameters(self):
        """
        Logs information about model parameters.
        """
        arg_params, aux_params = self.module.get_params()
        total_parameters = 0
        info = []  # type: List[str]
        for name, array in sorted(arg_params.items()):
            info.append("%s: %s" % (name, array.shape))
            total_parameters += reduce(lambda x, y: x * y, array.shape)
        logger.info("Model parameters: %s", ", ".join(info))
        if self.fixed_param_names:
            logger.info("Fixed model parameters: %s", ", ".join(self.fixed_param_names))
        logger.info("Total # of parameters: %d", total_parameters)

    def save_params_to_file(self, fname: str):
        """
        Synchronizes parameters across devices, saves the parameters to disk, and updates self.params
        and self.aux_params.

        :param fname: Filename to write parameters to.
        """
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        self.aux_params = aux_params
        super().save_params_to_file(fname)

    def load_params_from_file(self, fname: str, allow_missing_params: bool = False):
        """
        Loads parameters from a file and sets the parameters of the underlying module and this model instance.

        :param fname: File name to load parameters from.
        :param allow_missing_params: If set, the given parameters are allowed to be a subset of the Module parameters.
        """
        super().load_params_from_file(fname)  # sets self.params & self.aux_params
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=allow_missing_params)

    def install_monitor(self, monitor_pattern: str, monitor_stat_func_name: str):
        """
        Installs an MXNet monitor onto the underlying module.

        :param monitor_pattern: Pattern string.
        :param monitor_stat_func_name: Name of monitor statistics function.
        """
        self._monitor = mx.monitor.Monitor(interval=C.MEASURE_SPEED_EVERY,
                                           stat_func=C.MONITOR_STAT_FUNCS.get(monitor_stat_func_name),
                                           pattern=monitor_pattern,
                                           sort=True)
        self.module.install_monitor(self._monitor)
        logger.info("Installed MXNet monitor; pattern='%s'; statistics_func='%s'",
                    monitor_pattern, monitor_stat_func_name)

    @property
    def monitor(self) -> Optional[mx.monitor.Monitor]:
        return self._monitor
