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

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print('==> done')
    
    return model

def inference(model, test_set, opt):   
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    start = datetime.now()
    
    validate(test_set, model, criterion, opt)
            
    end = datetime.now()

    # calculate total time
    time_delta = round((end - start).total_seconds(), 3)
    print(f'Total inference time\t: {time_delta} seconds')
    print(f'Latency per image\t: {round(time_delta / 100, 3)} seconds')
    
def main(opt):
    print(f'model: {get_teacher_name(opt.path)}')
    
    test_loader, _ = get_cifar100_dataloaders_test()
    model = load_teacher(opt.path, 100)
    
    inference(model, test_loader, opt)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='Model path to evaluate')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')

    opt = parser.parse_args()
    print(opt)

    main(opt)