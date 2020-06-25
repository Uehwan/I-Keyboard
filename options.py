# Borrowed from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Modified by Uehwan Kim

import argparse
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')

        self.parser = parser
        self.arg_parsed = False

    def parse(self):
        # get the basic options
        if not self.arg_parsed:
            opt = self.parser.parse_args()
            self.opt = opt
            self.arg_parsed = True
        self.print_options(self.opt)

        return self.opt

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

        # save to the disk
        '''
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        '''


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--model', type=str, default='transformer',
                                 help='chooses which model to use. [unidirectional | bidirectional |'
                                      ' hierarchical | transformer | seq2seq_attention]')
        self.parser.add_argument('--embedding', dest='embedding', action='store_true',
                                 help='use char-embedding for seq2seq attention model')
        self.parser.add_argument('--no_embedding', dest='embedding', action='store_false',
                                 help='not to use char-embedding for seq2seq attention model')
        self.parser.set_defaults(embedding=True)
        self.parser.add_argument('--aux_sup', dest='aux_sup', action='store_true',
                                 help='use auxiliary supervision for the middle output')
        self.parser.add_argument('--no_aux_sup', dest='aux_sup', action='store_false',
                                 help='not to use auxiliary supervision for the middle output')
        self.parser.set_defaults(aux_sup=True)
        self.parser.add_argument('--is_training', dest='is_training', action='store_true',
                                 help='set the mode of the model for training')
        self.parser.add_argument('--is_testing', dest='is_training', action='store_false',
                                 help='set the mode of the model for testing')
        self.parser.set_defaults(is_training=True)
        self.parser.add_argument('--max_len', type=int, default=256,
                                 help='max sequence length; needed for position encoding')
        self.parser.add_argument('--embedding_size', type=int, default=16,
                                 help='size of character embedding vector')
        self.parser.add_argument('--num_layer', type=int, default=2,
                                 help='number of layers for each rnn')
        self.parser.add_argument('--num_unit', type=int, default=32,
                                 help='size of rnn cell')
        self.parser.add_argument('--num_head', type=int, default=8,
                                 help='# of heads for multi-head attention')
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='size of each batch for training')
        self.parser.add_argument('--rnn_type', type=str, default='GRU',
                                 help='which type of rnn to use. [LSTM | GRU]')
        self.parser.add_argument('--lr', type=float, default=0.001,
                                 help='initial learning rate for adam')
        self.parser.add_argument('--lr_decay', type=float, default=0.9,
                                 help='decay rate for learning rate')
        self.parser.add_argument('--num_epoch', type=int, default=500,
                                 help='# of epochs to train')
        self.parser.add_argument('--num_epoch_decay', type=int, default=500,
                                 help='# of epochs to linearly decay learning rate to zero')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on the console')
