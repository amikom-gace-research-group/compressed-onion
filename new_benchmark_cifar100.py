import torch
from torch import nn

import psutil
import os

from datetime import datetime

import CONFIG

def get_teacher_name(model_path):
    """parse teacher name"""
    model_segments = model_path.split('/')[-1].split('_')
    
    if model_segments[-1][:-4] == 'pruned':
        return model_segments[0], model_segments[1],model_segments[-3], model_segments[-2],model_segments[-1][:-4]
    elif model_segments[0] == 'wrn':
        return model_segments[0] + '_' + model_segments[1] + '_' + model_segments[2], model_segments[-1]
    else:
        return model_segments[0], model_segments[-1][:-4]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    # model_t = get_teacher_name(model_path)[0]
    # model = model_dict[model_t](num_classes=n_cls)
    model=torch.load(model_path)['model']
    model.to(CONFIG.DEVICE) #'cuda' if torch.cuda.is_available() else 'cpu' -> using cpu for experiment
    
    print('==> done')
    
    return model

def main(opt):
    p = psutil.Process(os.getpid())
    
    latency_list = []
    acc_list = []
    cpu_percent_list = []
    mem_percent_list = []
    # cpu_freq_list = []
    mem_use_list = []
    # cuda_power_list = []
    # cuda_percent_list = []
    # cuda_freq_list = []
    # cuda_mem_list = []