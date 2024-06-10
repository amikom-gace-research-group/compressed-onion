from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import scipy

from models import model_dict

def get_data_folder():

    data_folder = './data/flower102/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

def get_flowers102_dataloaders(batch_size=64, num_workers=8, is_instance=False):
    '''
    Flowers102
    '''
    
    data_folder = get_data_folder()
    transform = 
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    
    #WIP