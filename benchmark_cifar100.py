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

import psutil
import pandas as pd

# Get model name from the file name
def get_teacher_name(model_path):
    """parse teacher name"""
    model_segments = model_path.split('/')[-1].split('_')
    version_segment = model_path.split('/')[-2].split('_')
    
    if model_segments[0] != 'wrn':
        return model_segments[0], version_segment[-1]
    else:
        return model_segments[0] + '_' + model_segments[1] + '_' + model_segments[2], version_segment[-1]

#Loads weight with necessary model
def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)[0]
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
    latency_list = []
    acc_list = []
    cpu_percent_list = []
    mem_percent_list = []
    cpu_freq_list = []
    mem_use_list = []
    
    model_name = get_teacher_name(opt.path)
    
    print(f'model: {model_name}')
    
    test_loader, n_data = get_cifar100_dataloaders_test()
    model = load_teacher(opt.path, 100)
    
    for i in range(opt.iter):
        print(f'loop {i}')
        
        latency_data, acc_data = inference(model, test_loader, opt, n_data)
        
        cpu_percent_list.append(psutil.cpu_percent(interval=None))
        mem_percent_list.append(psutil.virtual_memory().percent)
        
        cpu_freq_list.append(psutil.cpu_freq().current)
        mem_use_list.append(round(((psutil.virtual_memory().available) / 1024 ** 3), 3))
        
        latency_list.append(latency_data)
        acc_list.append(round(acc_data, 3))
    
    print(f'Total images\t\t: {n_data}')
    
    data = {
        'latency' : latency_list,
        'accuracy' : acc_list,
        'cpu_percent' : cpu_percent_list,
        'cpu_freq' : cpu_freq_list,
        'mem_percent' : mem_percent_list,
        'mem_use_MB' : mem_use_list
    }
    
    df = pd.DataFrame(data)
    
    if opt.save == True:
    
        if not os.path.isdir(f'./benchmark/result/{model_name[0]}'):
            os.makedirs(f'./benchmark/result/{model_name[0]}')
        
        df.to_csv(f'./benchmark/result/{model_name[0]}/{model_name[0]}_{model_name[1]}.csv')
        
    else:
        print(df)  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='Model path to evaluate')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--iter', type=int, default=60, help='Number of Itteration to execute')
    parser.add_argument('--save', type=bool, default=False, help='True - False, default is unsaved')

    opt = parser.parse_args()
    print(opt)

    main(opt)