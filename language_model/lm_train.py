from . import lm_model.TrainingLanguageModel
import argparse

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # TODO: add parameters
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    # TODO: perform training on lm_model.TrainingLanguageModel
