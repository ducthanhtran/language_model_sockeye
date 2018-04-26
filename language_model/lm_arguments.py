import argparse
import sys

sys.path.append('../')

from sockeye.arguments import regular_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_cli_data(parser)


    return parser


def add_cli_data(parser: arparse.ArgumentParser) -> None:
    parser.add_argument('--train-data',
                        required=True,
                        type=sockeye.arguments.regular_file(),
                        help='training data. Target labels are generated')
    parser.add_argument('--dev-data',
                        required=True,
                        type=sockeye.arguments.regular_file(),
                        help='development data - used for early stopping')

def add_cli_optimizer(parser: argparse.ArgumentParser) -> None:
    parser.add_argument()
