import time
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
from torchvision import datasets, transforms, models

import json
from PIL import Image
  
import argparse

from helper import loaders, run_the_model, train
## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()


    parser.add_argument('data_dir')    
    
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help="directory to save checkpoint example: dir/")
    
    parser.add_argument('--arch',
                        type=str,
                        default='vgg13',
                        help="CNN architecture (default: vgg13)")
    
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help="")
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        metavar='N',
                        help='number of epochs to train (default: 25)')
    
    parser.add_argument('--hidden_units',
                        type=int,
                        default=256,
                        metavar='S',
                        help='# of hidden units (default: 4)')

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        metavar='S',
                        help='# random seed (default: 1234)')
    
    parser.add_argument('--gpu',
                        type=bool,
                        default=False,
                        help='Indicator of using GPU (default:False)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = args.gpu
    print("Using device {}.".format("GPU-cuda" if use_cuda else "CPU"))

    torch.manual_seed(args.seed)
    print(args)
    # Load the training data.
    train_dataset, data_loaders = loaders(args.batch_size, args.data_dir)
    print(data_loaders)
   
    classifier, model = run_the_model(args.arch,
                                      args.hidden_units,
                                      len(train_dataset.classes))

    if use_cuda:
        model = model.cuda()
    
    criterion= nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
        
    trained_model = train(args.epochs,
                        data_loaders,
                        model,
                        optimizer,
                        criterion,
                        use_cuda,
                        'outputs/model_results.pt')
        
    if args.save_dir is not None:
        checkpoint = {'arch':args.arch,
                      'output_size':len(train_dataset.classes),
                      'classifier': classifier,
                      'model_state':trained_model.state_dict(),
                      'optimizer':optimizer.state_dict(),
                      'class_to_index':train_dataset.class_to_idx}
        torch.save(checkpoint,args.save_dir+"checkpoint.pth")
        print("Checkpoint saved!")

