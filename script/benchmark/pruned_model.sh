#!/bin/bash

models=(
    "vgg8"
    "vgg11"
    "vgg13"
    "vgg16"
    "vgg19"
)

sparcity=(
    "0.1"
    "0.05"
    "0.01"
)

for model in "${models[@]}"
do
    for num in "${sparcity[@]}"
    do
        path="save/models/${model}/${model}_l1_Complete_${num}_pruned.pth"
        python3 -m benchmark_cifar100 --path "${path}" --save 1
    done
done