import torch
from torch import nn
from torch import optim
from PIL import Image

import CONFIG
from models import model_dict
from dataset import cifar100

import datetime
import argparse

model_path = ''

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    model.to(CONFIG.DEVICE)
    print('==> done')
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameter(), lr=.001)
    
    return model, loss_fn, optimizer

def inference(model, transform, image_files):
    model.eval()
    
    start = datetime.now()
    
    with torch.no_grad():
        for image_file in image_files:
            image = Image.open(image_file)
            image = transform(image).unsqueeze(0)  # Add batch dimension
            image = image.to(CONFIG.DEVICE)

            _ = model(image)
            
    end = datetime.now()

    # calculate total time
    time_delta = round((end - start).total_seconds(), 2)
    print(f'Total inference time\t: {time_delta} seconds')
    print(f'Latency per image\t: {round(time_delta / 100, 2)} seconds')
    
def main(args):
    pass
