import torch
from torch import nn
from torch import optim
from PIL import Image

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders_test
from helper.loops import validate

import torch.backends.cudnn as cudnn

from datetime import datetime
import argparse

import numpy as np
import os

# Get model name from the file name
def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

#Loads weight with necessary model
def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('==> done')
    
    return model

def inference(model, test_set, opt, n_data):   
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    start = datetime.now()
    
    #! Not needed tp model.eval() validate func already have it!    
    test_acc, _, _ = validate(test_set, model, criterion, opt)
            
    end = datetime.now()

    # calculate total time
    time_delta = round((end - start).total_seconds(), 3)
    print(f'Total inference time\t: {time_delta} seconds') #in seconds
    print(f'Latency per image\t: {round((time_delta/ n_data)*1000, 4)} ms') #in miliseconds
    
    return time_delta, test_acc.item()
    
    
    
def main(opt):
    meta_arr = []
    
    model_name = get_teacher_name(opt.path)
    
    print(f'model: {model_name}')
    
    test_loader, n_data = get_cifar100_dataloaders_test()
    model = load_teacher(opt.path, 100)
    
    for i in range(opt.iter):
        print(f'loop {i}')
        
        latency_data, acc_data = inference(model, test_loader, opt, n_data)
        
        meta_arr.append([latency_data, round(acc_data, 3)])
    
    print(f'Total images\t\t: {n_data}')
    
    if opt.save != '-':
    
        if not os.path.isdir(f'./benchmark/result/{model_name}'):
            os.makedirs(f'./benchmark/result/{model_name}')
        
        np.savetxt(f'./benchmark/result/{model_name}/{model_name}_{opt.save}.csv',
                meta_arr,
                delimiter=',',
                fmt='% s')
    else:
        print(meta_arr)  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='Model path to evaluate')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--iter', type=int, default=60, help='Number of Itteration to execute')
    parser.add_argument('--save', type=str, default='-', help='File name for storing benchmark data, dafault is unsaved')

    opt = parser.parse_args()
    print(opt)

    main(opt)