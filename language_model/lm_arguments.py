import argparse
import sys

sys.path.append('../')

import sockeye.constants as C
from sockeye.arguments import regular_file, regular_folder, simple_dict, int_greater_or_equal, learning_schedule


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_params_data(parser)
    add_params_output(parser)
    add_params_bucketing(parser)
    add_params_device(parser)
    add_params_vocab(parser)
    add_params_model(parser)
    add_params_training(parser)
    return parser


def add_params_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--train-data',
                        required=True,
                        type=regular_file(),
                        help='training data. Target labels are generated')
    parser.add_argument('--dev-data',
                        required=True,
                        type=regular_file(),
                        help='development data - used for early stopping')


def add_params_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--output', '-o',
                        required=True,
                        help='Folder where model & training results are written to.')
    parser.add_argument('--overwrite-output',
                        action='store_true',
                        help='Delete all contents of the model directory if it already exists.')


def add_params_bucketing(parser: argparse.ArgumentParser) -> None:
    bucketing_params = parser.add_argument_group('Bucketing parameters')
    bucketing_params.add_argument('--no-bucketing',
                        action='store_true',
                        help='Disable bucketing: always unroll the graph to --max-seq-len. Default: %(default)s.')

    bucketing_params.add_argument('--bucket-width',
                        type=int_greater_or_equal(1),
                        default=10,
                        help='Width of buckets in tokens. Default: %(default)s.')

    bucketing_params.add_argument('--max-seq-len',
                        type=int_greater_or_equal(1),
                        default=100,
                        help='Maximum sequence length in tokens. Note that the target side will be extended by '
                             'the <BOS> (beginning of sentence) token, increasing the effective target length. '
                             'Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')


def add_params_device(parser: argparse.ArgumentParser) -> None:
    device_params = parser.add_argument_group("Device parameters")

    device_params.add_argument('--device-ids', default=[-1],
                               help='List or number of GPUs ids to use. Default: %(default)s. '
                                    'Use negative numbers to automatically acquire a certain number of GPUs, e.g. -5 '
                                    'will find 5 free GPUs. '
                                    'Use positive numbers to acquire a specific GPU id on this host. '
                                    '(Note that automatic acquisition of GPUs assumes that all GPU processes on '
                                    'this host are using automatic sockeye GPU acquisition).',
                               nargs='+', type=int)
    device_params.add_argument('--use-cpu',
                               action='store_true',
                               help='Use CPU device instead of GPU.')
    device_params.add_argument('--disable-device-locking',
                               action='store_true',
                               help='Just use the specified device ids without locking.')
    device_params.add_argument('--lock-dir',
                               default="/tmp",
                               help='When acquiring a GPU we do file based locking so that only one Sockeye process '
                                    'can run on the a GPU. This is the folder in which we store the file '
                                    'locks. For locking to work correctly it is assumed all processes use the same '
                                    'lock directory. The only requirement for the directory are file '
                                    'write permissions.')


def add_params_vocab(params):
    params.add_argument('--vocab',
                        required=False,
                        default=None,
                        help='Existing source vocabulary (JSON).')
    params.add_argument('--num-words',
                        type=int_greater_or_equal(0),
                        default=50000,
                        help='Maximum vocabulary size. Use "x:x" to specify separate values for src&tgt. '
                             'Default: %(default)s.')
    params.add_argument('--word-min-count',
                        type=int_greater_or_equal(1),
                        default=1,
                        help='Minimum frequency of words to be included in vocabularies. Default: %(default)s.')


def add_params_model(parser: argparse.ArgumentParser):
    model_params = parser.add_argument_group("ModelConfig")

    model_params.add_argument('--params', '-p',
                              type=str,
                              default=None,
                              help='Initialize model parameters from file. Overrides random initializations.')
    model_params.add_argument('--allow-missing-params',
                              action="store_true",
                              default=False,
                              help="Allow missing parameters when initializing model parameters from file. "
                                   "Default: %(default)s.")

    model_params.add_argument('--num-layers',
                              type=int_greater_or_equal(1),
                              default=1,
                              help='Number of layers for stacked RNN language model. Default: %(default)s.')

    # rnn arguments
    model_params.add_argument('--rnn-cell-type',
                              choices=C.CELL_TYPES,
                              default=C.LSTM_TYPE,
                              help='RNN cell type for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-num-hidden',
                              type=int_greater_or_equal(1),
                              default=1024,
                              help='Number of RNN hidden units for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-residual-connections',
                              action="store_true",
                              default=False,
                              help="Add residual connections to stacked RNNs. (see Wu ETAL'16). Default: %(default)s.")
    model_params.add_argument('--rnn-first-residual-layer',
                              type=int_greater_or_equal(2),
                              default=2,
                              help='First RNN layer to have a residual connection. Default: %(default)s.')
    model_params.add_argument('--rnn-context-gating', action="store_true",
                              help="Enables a context gate which adaptively weighs the RNN decoder input against the "
                                   "source context vector before each update of the decoder hidden state.")

    # embedding arguments
    model_params.add_argument('--num-embed',
                              type=int_greater_or_equal(1),
                              default=512,
                              help='Embedding size for tokens. Default: %(default)s.')
    model_params.add_argument('--weight-tying',
                              action='store_true',
                              help='Turn on weight tying (see arxiv.org/abs/1608.05859). '
                                   'Default: False.')

    model_params.add_argument('--layer-normalization', action="store_true",
                              help="Adds layer normalization before non-linear activations. "
                                   "This includes MLP attention, RNN decoder state initialization, "
                                   "RNN decoder hidden state, and cnn layers."
                                   "It does not normalize RNN cell activations "
                                   "(this can be done using the '%s' or '%s' rnn-cell-type." % (C.LNLSTM_TYPE,
                                                                                                C.LNGLSTM_TYPE))
    model_params.add_argument('--weight-normalization', action="store_true",
                              help="Adds weight normalization to decoder output layers "
                                   "(and all convolutional weight matrices for CNN decoders). Default: %(default)s.")


def add_params_training(params):
    train_params = params.add_argument_group("Training parameters")

    train_params.add_argument('--batch-size', '-b',
                              type=int_greater_or_equal(1),
                              default=64,
                              help='Mini-batch size. Default: %(default)s.')
    train_params.add_argument("--batch-type",
                              type=str,
                              default=C.BATCH_TYPE_SENTENCE,
                              choices=[C.BATCH_TYPE_SENTENCE, C.BATCH_TYPE_WORD],
                              help="Sentence: each batch contains X sentences, number of words varies. Word: each batch"
                                   " contains (approximately) X words, number of sentences varies. Default: %(default)s.")

    train_params.add_argument('--fill-up',
                              type=str,
                              default='replicate',
                              help=argparse.SUPPRESS)

    train_params.add_argument('--loss',
                              default=C.CROSS_ENTROPY,
                              choices=[C.CROSS_ENTROPY],
                              help='Loss to optimize. Default: %(default)s.')
    train_params.add_argument('--label-smoothing',
                              default=0.0,
                              type=float,
                              help='Smoothing constant for label smoothing. Default: %(default)s.')
    train_params.add_argument('--loss-normalization-type',
                              default=C.LOSS_NORM_VALID,
                              choices=[C.LOSS_NORM_VALID, C.LOSS_NORM_BATCH],
                              help='How to normalize the loss. By default we normalize by the number '
                                   'of valid/non-PAD tokens (%s)' % C.LOSS_NORM_VALID)

    train_params.add_argument('--metrics',
                              nargs='+',
                              default=[C.PERPLEXITY],
                              choices=[C.PERPLEXITY, C.ACCURACY],
                              help='Names of metrics to track on training and validation data. Default: %(default)s.')
    train_params.add_argument('--optimized-metric',
                              default=C.PERPLEXITY,
                              choices=C.METRICS,
                              help='Metric to optimize with early stopping {%(choices)s}. '
                                   'Default: %(default)s.')

    train_params.add_argument('--max-updates',
                              type=int,
                              default=None,
                              help='Maximum number of updates/batches to process. Default: %(default)s.')
    train_params.add_argument(C.TRAIN_ARGS_CHECKPOINT_FREQUENCY,
                              type=int_greater_or_equal(1),
                              default=1000,
                              help='Checkpoint and evaluate every x updates/batches. Default: %(default)s.')
    train_params.add_argument('--max-num-checkpoint-not-improved',
                              type=int,
                              default=8,
                              help='Maximum number of checkpoints the model is allowed to not improve in '
                                   '<optimized-metric> on validation data before training is stopped. '
                                   'Default: %(default)s')
    train_params.add_argument('--min-num-epochs',
                              type=int,
                              default=None,
                              help='Minimum number of epochs (passes through the training data) '
                                   'before fitting is stopped. Default: %(default)s.')
    train_params.add_argument('--max-num-epochs',
                              type=int,
                              default=None,
                              help='Maximum number of epochs (passes through the training data) '
                                   'before fitting is stopped. Default: %(default)s.')

    train_params.add_argument('--embed-dropout',
                              type=float,
                              default=.0,
                              help='Dropout probability for source & target embeddings. Default: %(default)s.')
    train_params.add_argument('--rnn-dropout-inputs',
                              type=float,
                              default=.0,
                              help='RNN variational dropout probability for language model decoder RNN inputs. '
                                   '(Gal, 2015). Default: %(default)s.')
    train_params.add_argument('--rnn-dropout-states',
                              type=float,
                              default=.0,
                              help='RNN variational dropout probability for decoder RNN states. (Gal, 2015). '
                                   'Default: %(default)s.')
    train_params.add_argument('--rnn-dropout-recurrent',
                              type=float,
                              default=.0,
                              help='Recurrent dropout without memory loss (Semeniuta, 2016) for decoder '
                                   'LSTMs. Default: %(default)s.')

    train_params.add_argument('--rnn-decoder-hidden-dropout',
                              type=float,
                              default=.0,
                              help='Dropout probability for hidden state that combines the context with the '
                                   'RNN hidden state in the decoder. Default: %(default)s.')
    train_params.add_argument('--optimizer',
                              default=C.OPTIMIZER_ADAM,
                              choices=C.OPTIMIZERS,
                              help='SGD update rule. Default: %(default)s.')
    train_params.add_argument('--optimizer-params',
                              type=simple_dict(),
                              default=None,
                              help='Additional optimizer params as dictionary. Format: key1:value1,key2:value2,...')

    train_params.add_argument("--kvstore",
                              type=str,
                              default=C.KVSTORE_DEVICE,
                              choices=C.KVSTORE_TYPES,
                              help="The MXNet kvstore to use. 'device' is recommended for single process training. "
                                   "Use any of 'dist_sync', 'dist_device_sync' and 'dist_async' for distributed "
                                   "training. Default: %(default)s.")
    train_params.add_argument("--gradient-compression-type",
                              type=str,
                              default=C.GRADIENT_COMPRESSION_NONE,
                              choices=C.GRADIENT_COMPRESSION_TYPES,
                              help='Type of gradient compression to use. Default: %(default)s.')
    train_params.add_argument("--gradient-compression-threshold",
                              type=float,
                              default=0.5,
                              help="Threshold for gradient compression if --gctype is '2bit'. Default: %(default)s.")

    train_params.add_argument('--weight-init',
                              type=str,
                              default=C.INIT_XAVIER,
                              choices=C.INIT_TYPES,
                              help='Type of base weight initialization. Default: %(default)s.')
    train_params.add_argument('--weight-init-scale',
                              type=float,
                              default=2.34,
                              help='Weight initialization scale. Applies to uniform (scale) and xavier (magnitude). '
                                   'Default: %(default)s.')
    train_params.add_argument('--weight-init-xavier-factor-type',
                              type=str,
                              default='in',
                              choices=['in', 'out', 'avg'],
                              help='Xavier factor type. Default: %(default)s.')
    train_params.add_argument('--weight-init-xavier-rand-type',
                              type=str,
                              default=C.RAND_TYPE_UNIFORM,
                              choices=[C.RAND_TYPE_UNIFORM, C.RAND_TYPE_GAUSSIAN],
                              help='Xavier random number generator type. Default: %(default)s.')
    train_params.add_argument('--embed-weight-init',
                              type=str,
                              default=C.EMBED_INIT_DEFAULT,
                              choices=C.EMBED_INIT_TYPES,
                              help='Type of embedding matrix weight initialization. If normal, initializes embedding '
                                   'weights using a normal distribution with std=1/srqt(vocab_size). '
                                   'Default: %(default)s.')
    train_params.add_argument('--initial-learning-rate',
                              type=float,
                              default=0.0003,
                              help='Initial learning rate. Default: %(default)s.')
    train_params.add_argument('--weight-decay',
                              type=float,
                              default=0.0,
                              help='Weight decay constant. Default: %(default)s.')
    train_params.add_argument('--momentum',
                              type=float,
                              default=None,
                              help='Momentum constant. Default: %(default)s.')
    train_params.add_argument('--gradient-clipping-threshold',
                              type=float,
                              default=1.0,
                              help='Clip absolute gradients values greater than this value. '
                                   'Set to negative to disable. Default: %(default)s.')
    train_params.add_argument('--gradient-clipping-type',
                              choices=C.GRADIENT_CLIPPING_TYPES,
                              default=C.GRADIENT_CLIPPING_TYPE_ABS,
                              help='The type of gradient clipping. Default: %(default)s.')

    train_params.add_argument('--learning-rate-scheduler-type',
                              default=C.LR_SCHEDULER_PLATEAU_REDUCE,
                              choices=C.LR_SCHEDULERS,
                              help='Learning rate scheduler type. Default: %(default)s.')
    train_params.add_argument('--learning-rate-reduce-factor',
                              type=float,
                              default=0.5,
                              help="Factor to multiply learning rate with "
                                   "(for 'plateau-reduce' learning rate scheduler). Default: %(default)s.")
    train_params.add_argument('--learning-rate-reduce-num-not-improved',
                              type=int,
                              default=3,
                              help="For 'plateau-reduce' learning rate scheduler. Adjust learning rate "
                                   "if <optimized-metric> did not improve for x checkpoints. Default: %(default)s.")
    train_params.add_argument('--learning-rate-schedule',
                              type=learning_schedule(),
                              default=None,
                              help="For 'fixed-step' scheduler. Fully specified learning schedule in the form"
                                   " \"rate1:num_updates1[,rate2:num_updates2,...]\". Overrides all other args related"
                                   " to learning rate and stopping conditions. Default: %(default)s.")
    train_params.add_argument('--learning-rate-half-life',
                              type=float,
                              default=10,
                              help="Half-life of learning rate in checkpoints. For 'fixed-rate-*' "
                                   "learning rate schedulers. Default: %(default)s.")
    train_params.add_argument('--learning-rate-warmup',
                              type=int,
                              default=0,
                              help="Number of warmup steps. If set to x, linearly increases learning rate from 10%% "
                                   "to 100%% of the initial learning rate. Default: %(default)s.")
    train_params.add_argument('--learning-rate-decay-param-reset',
                              action='store_true',
                              help='Resets model parameters to current best when learning rate is reduced due to the '
                                   'value of --learning-rate-reduce-num-not-improved. Default: %(default)s.')
    train_params.add_argument('--learning-rate-decay-optimizer-states-reset',
                              choices=C.LR_DECAY_OPT_STATES_RESET_CHOICES,
                              default=C.LR_DECAY_OPT_STATES_RESET_OFF,
                              help="Action to take on optimizer states (e.g. Adam states) when learning rate is "
                                   "reduced due to the value of --learning-rate-reduce-num-not-improved. "
                                   "Default: %(default)s.")

    train_params.add_argument('--rnn-forget-bias',
                              default=0.0,
                              type=float,
                              help='Initial value of RNN forget biases.')
    train_params.add_argument('--rnn-h2h-init', type=str, default=C.RNN_INIT_ORTHOGONAL,
                              choices=[C.RNN_INIT_ORTHOGONAL, C.RNN_INIT_ORTHOGONAL_STACKED, C.RNN_INIT_DEFAULT],
                              help="Initialization method for RNN parameters. Default: %(default)s.")

    train_params.add_argument(C.TRAIN_ARGS_MONITOR_BLEU,
                              default=0,
                              type=int,
                              help='x>0: decode x sampled sentences from validation data and '
                                   'compute evaluation metrics. x==-1: use full validation data. Default: %(default)s.')
    train_params.add_argument('--seed',
                              type=int,
                              default=13,
                              help='Random seed. Default: %(default)s.')

    train_params.add_argument('--keep-last-params',
                              type=int,
                              default=-1,
                              help='Keep only the last n params files, use -1 to keep all files. Default: %(default)s')

    train_params.add_argument('--dry-run',
                              action='store_true',
                              help="Do not perform any actual training, but print statistics about the model"
                              " and mode of operation.")
