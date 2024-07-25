import torch
import torch_pruning as tp
import torch.nn as nn

import argparse
import os

import pandas as pd

from models import model_dict

def get_model_name(model_path):
    """parse model name"""
    model_segments = model_path.split('/')[-1].split('_')
    version_segment = model_path.split('/')[-2].split('_')
    
    if model_segments[0] != 'wrn':
        return model_segments[0], version_segment[1]
    else:
        return model_segments[0] + '_' + model_segments[1] + '_' + model_segments[2], version_segment[1]

def main(args):
    model_path = args.path
    prune_method = args.prune_method
    ratio = args.ratio
    global_pruning = args.global_pruning

    model_name = get_model_name(model_path=model_path)
    model = model_dict[model_name[0]](num_classes=100)
    model.load_state_dict(torch.load(model_path)['model'])

    example_inputs = torch.randn(1, 3, 32, 32) #CIFAR 100 image size

    # 1. Importance criterion
    if prune_method == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    elif prune_method == 'l2':
        imp = tp.importance.MagnitudeImportance(p=2)
    elif prune_method == 'random':
        imp = tp.importance.RandomImportance()
    else:
        NotImplementedError

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 100:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
        global_pruning=global_pruning
    )

    # 3. Prune & finetune the model
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f'base MACs: {base_macs/1e6:.2f} M \t base_params: {base_nparams/1e6:.2f} M')

    if isinstance(imp, tp.importance.GroupTaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        loss = nn.CrossEntropyLoss() # A dummy loss, please replace this line with your loss function and data!
        loss.backward() # before pruner.step()

    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f'pruned MACs: {macs/1e6:.2f} M \t pruned params: {nparams/1e6:.2f} M')

    # save dir      
    file_name = f'{model_name[0]}_{model_name[1]}_prune_{prune_method}_{ratio}_{args.id}.pth'
    save_path = os.path.join(args.save, model_name[0])
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    save_path = os.path.join(save_path, file_name)
    
    state={
        'model' : model
    }
    
    torch.save(state, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,required=True)
    parser.add_argument('--prune-method', type=str, required=True, choices=['l1', 'l2', 'random'])
    parser.add_argument('--ratio', type=float, default=.5, required=True)
    parser.add_argument('--global-pruning', default=False)
    parser.add_argument('--save', type=str, default='save/pruned/')
    parser.add_argument('--id', type=str, default='none')
    args = parser.parse_args()

    main(args)