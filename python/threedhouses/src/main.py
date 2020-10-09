'''
main.py

Welcome, this is the entrance to 3dgan
'''

import argparse
from trainer import *
import torch

from tester import *
from osl_learning import *
import params

def main():

    # add arguments
    parser = argparse.ArgumentParser()

    # loggings parameters
    parser.add_argument('--logs', action='store_true', help='logs by tensorboardX')
    parser.add_argument('--model_name', type=str, default="dcgan", help='model name for saving')
    parser.add_argument('--test', action='store_true', help='call tester.py')
    parser.add_argument('--iter', action='store_true', help='call tester.py')
    parser.add_argument('--use_visdom', action='store_true', help='visualization by visdom')
    #parser.add_argument('--ae', action='store_true', help='train auto-encoder')
    parser.add_argument('--debug', action='store_true', help='Useful for debugging: 0 thread data loader')
    parser.add_argument('--probe', action='store_true', help='Analyze probe metrics')

    args = parser.parse_args()

    # list params
    params.print_params()

    # run program
    if args.test:
        tester_ae(args)
    elif args.probe:
        #tester_embedding_probe(args)
        tester_generate_osl_bank(args)
    elif args.iter:
        iterate_through_dataset(args)
    else:
        trainer_autoencoder_bl(args)

if __name__ == '__main__':
    main()

    
