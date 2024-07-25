import torch
from torchsummary import summary

import argparse

from models import model_dict
import CONFIG

def get_teacher_name(model_path):
    """parse teacher name"""
    model_segments = model_path.split('/')[-1].split('_')
    version_segment = model_path.split('/')[-2].split('_')
    
    if model_segments[0] != 'wrn':
        return model_segments[0], version_segment[-1]
    else:
        return model_segments[0] + '_' + model_segments[1] + '_' + model_segments[2], version_segment[-1]
    
def main(args):
    print('==> loading teacher model')
    # model_name = get_teacher_name(args.path)
    # model = model_dict[model_name[0]](num_classes=100)
    model = torch.load(args.path)['model']
    model.to(CONFIG.DEVICE)
    
    print(f'model path: {args.path}')
    model.eval()
    
    summary(model, (3, 32, 32))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    main(args)