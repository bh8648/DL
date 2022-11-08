# <학습자료 8-1>

# define_argparser 함수
import argparse

import torch.cuda


def define_argparser() :
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=5)

    p. add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config


def main(config) :
    pass


