# RL_learning_rate
import argparse
import os
import torch
import models


class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='./MNIST_data',
                            help='path to images')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='./checkpoints', help='models are saved here')
        parser.add_argument('--seed', type=int, default=3, help='random seed')
        parser.add_argument('--batch_size', type=int,
                            default=3000, help='input batch size')
        parser.add_argument('--use_existing_classification_model', type=bool,
                            default=True, help='use existing classification model')

        # model parameters
        parser.add_argument('--cnn_learning_rate', type=float,
                            default=0.005, help='learning rate for cnn')
        parser.add_argument('--rl_learning_rate', type=float,
                            default=0.0001, help='learning rate for RL')
        parser.add_argument('--num_epochs', type=int,  default=100,
                            help='number of epochs to train for')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()

        # process opt.suffix
        self.print_options(opt)
        self.opt = opt
        return self.opt
