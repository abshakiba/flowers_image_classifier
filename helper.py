import time
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms, models

import json
from PIL import Image


def loaders(batch_size, data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.Resize(258),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           normalize])

    valid_transforms = transforms.Compose([transforms.Resize(258),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


    loaders = {
        'train': trainloader,
        'valid': validloader,
    }
    return train_dataset, loaders

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""

    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        
        train_loss = 0.0
        valid_loss = 0.0
        
        model.train()
        
        for batch_idx, (data, target) in enumerate(loaders['train']):
            
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            optimizer.zero_grad()
            
            output = model(data)
            
            loss = criterion(output, target)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
#             print("Running Train Loss:",train_loss)

        model.eval()
        
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            ## update the average validation loss
            output = model(data)   
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: >>>{:.6f} \tValidation Loss: >>>{:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        if valid_loss < valid_loss_min:
            print('Saved model with valid_loss:{:.6f}...'.format(valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss            

    return model

def run_the_model(arch, hidden_units, num_classes):
    if arch=="vgg13":
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.25),
                           nn.Linear(hidden_units, int(hidden_units/2)),
                           nn.ReLU(),
                           nn.Dropout(0.25),
                           nn.Linear(int(hidden_units/2), num_classes)) 
        model.classifier = classifier
    return classifier, model

def load_checkpoint(path, device):
    
    checkpoint = torch.load(path)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_index']
    
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    use_cuda = torch.cuda.is_available()
    
    model = model.to(device)
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    pred_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])
    

    return pred_transform(img)

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path).unsqueeze(0).to(device)

    model = model.to(device)
    
    model.eval()    
    
    with torch.no_grad():

        output = F.log_softmax(model(image), dim=1)
        output = torch.exp(output)
        top_prob, top_labels = torch.topk(output, topk, dim=1)
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes =[]

    for label in top_labels.cpu().numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.cpu().numpy()[0], mapped_classes

def cat_to_name(image_path, cat_code, category_file):
    actual_code = image_path.split("/")[-2]
    if category_file is not None:
        with open(category_file, 'r') as f:
            cat_to_name = json.load(f)
        classes = []
        for idx in cat_code:
            classes.append(cat_to_name[idx])
        return cat_to_name[actual_code], classes
    return actual_code ,cat_code