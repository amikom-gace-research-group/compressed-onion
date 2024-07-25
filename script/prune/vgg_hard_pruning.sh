#!/bin/bash

models=(
    "vgg8"
    "vgg11"
    "vgg13"
    "vgg16"
    "vgg19"
)

ratios=(
    "0.2"
    "0.5"
    "0.8"
)

methods=(
    "l1"
    "l2"
    "random"
)



for model in "${models[@]}"
do
    for ratio in "${ratios[@]}"
    do
        for method in "${methods[@]}"
        do
            path="save/state_dict/models/${model}/${model}_cifar100_lr_0.05_decay_0.0005_trial_base/${model}_last.pth"
            python3 -m hard_pruning --path "${path}" --prune-method "${method}" --ratio "${ratio}" --global-pruning "True"
        done
    done
done