import torch
from torch import nn, optim, cuda
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

import CONFIG

# Get model name from the file name
def get_teacher_name(model_path):
    """parse teacher name"""
    model_segments = model_path.split('/')[-1].split('_')
    
    if model_segments[-1][:-4] == 'pruned':
        return model_segments[0], model_segments[-3], model_segments[-2],model_segments[-1][:-4]
    elif model_segments[0] == 'wrn':
        return model_segments[0] + '_' + model_segments[1] + '_' + model_segments[2], model_segments[-1]
    else:
        return model_segments[0], model_segments[-1][:-4]

#Loads weight with necessary model
def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)[0]
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    model.to(CONFIG.DEVICE) #'cuda' if torch.cuda.is_available() else 'cpu' -> using cpu for experiment
    
    print('==> done')
    
    return model

def inference(model, test_set, opt, n_data):   
    criterion = nn.CrossEntropyLoss()

    # TODO: Change this code below and other to xxx.to(CONFIG.DEVICE)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     criterion = criterion.cuda()
    #     cudnn.benchmark = True
    
    model = model.to(CONFIG.DEVICE)
    criterion = criterion.to(CONFIG.DEVICE)
    
    start = datetime.now()
    
    #! Not needed to model.eval() validate func already have it!    
    test_acc, _, _ = validate(test_set, model, criterion, opt)
            
    end = datetime.now()

    # calculate total time
    time_delta = round((end - start).total_seconds(), 3)
    print(f'Total inference time\t: {time_delta} seconds') #in seconds
    print(f'Latency per image\t: {round((time_delta/ n_data)*1000, 4)} ms') #in miliseconds
    
    return time_delta, test_acc.item()
    
    
    
def main(opt):
    p = psutil.Process(os.getpid())
    
    latency_list = []
    acc_list = []
    cpu_percent_list = []
    mem_percent_list = []
    # cpu_freq_list = []
    mem_use_list = []
    cuda_power_list = []
    cuda_percent_list = []
    cuda_freq_list = []
    cuda_mem_list = []
    
    
    
    model_name = get_teacher_name(opt.path)
    
    print(f'model: {model_name}')
    
    test_loader, n_data = get_cifar100_dataloaders_test()
    model = load_teacher(opt.path, 100)
    
    for i in range(opt.iter + 1): # Added 1 for removed 1 row of data later
        print(f'loop {i}')
        
        latency_data, acc_data = inference(model, test_loader, opt, n_data)
        
        cpu_percent_list.append(round(p.cpu_percent(interval=None)/psutil.cpu_count(), 2))
        mem_percent_list.append(round(p.memory_percent(), 2))
        
        # cpu_freq_list.append(round(p.cpu, 3))
        mem_use_list.append(round(((p.memory_full_info().vms) / 1024 ** 3), 2)) # In MB
        
        cuda_power_list.append(round(cuda.power_draw()/1000, 2)) # In mW, converted to Watts
        cuda_percent_list.append(cuda.utilization()) # In %
        cuda_mem_list.append(cuda.memory_usage()) # In %
        cuda_freq_list.append(cuda.clock_rate()) # In Hz 
        
        latency_list.append(latency_data)
        acc_list.append(round(acc_data, 3))
    
    print(f'Total images\t\t: {n_data}')
    
    # '[1:]' is used to get rid warmup outliar
    data = {
        'latency' : latency_list[1:],
        'accuracy' : acc_list[1:],
        'cpu_percent' : cpu_percent_list[1:],
        # 'cpu_freq' : cpu_freq_list[1:],
        'mem_percent' : mem_percent_list[1:],
        'mem_use_MB' : mem_use_list[1:],
        'gpu_percent' : cuda_percent_list[1:],
        'gpu_freq' : cuda_freq_list[1:],
        'gpu_mem_percent' : cuda_mem_list[1:],
        'gpu_power_w' : cuda_power_list[1:]
    }
    
    df = pd.DataFrame(data)
    
    if opt.save == 1:
    
        if not os.path.isdir(f'./benchmark/result/{model_name[0]}'):
            os.makedirs(f'./benchmark/result/{model_name[0]}')
        
        if model_name[1] != 'last':
            df.to_csv(f'./benchmark/result/{model_name[0]}/{model_name[0]}_{model_name[1]}_{model_name[2]}_{model_name[3]}.csv')
        else:
            df.to_csv(f'./benchmark/result/{model_name[0]}/{model_name[0]}_{model_name[1]}.csv')
        
    else:
        print(df)  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='Model path to evaluate')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--iter', type=int, default=60, help='Number of Itteration to execute')
    parser.add_argument('--save', type=int, default=0, help='True(1) - False(0), default is unsaved and printed on CLI')
    
    opt = parser.parse_args()
    print(opt)

    main(opt)
    
    print(f'Model Tested : {opt.path}')
    print('Harware Used : {hardware}'.format(hardware=CONFIG.DEVICE))