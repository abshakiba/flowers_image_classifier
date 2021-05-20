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

from helper import load_checkpoint, process_image, predict, cat_to_name
## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()


    parser.add_argument('image_path')
    
    parser.add_argument('checkpoint') 
    
    parser.add_argument('--topk',
                        type=int,
                        metavar="K",
                        default=5)
    
    parser.add_argument('--category_names',
                        type=str,
                        default=None,
                        help="Category to name, json file")

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

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    
#     print(args)
    
    model = load_checkpoint(args.checkpoint, device)
    probs, pred_classes = predict(args.image_path, model, device, args.topk)
    
    actual_class, pred_classes = cat_to_name(args.image_path, pred_classes, args.category_names)
    print(actual_class)
    print(probs, pred_classes)