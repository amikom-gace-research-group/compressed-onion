# Compressed Onion
Experimenting with Pruned Teachers on KD compression method

This code was referenced from [RepDistiller](https://github.com/HobbitLong/RepDistiller) and [gace-characterize-pruning](https://github.com/amikom-gace-research-group/characterize-pruning) repo.

## Setup
This experiment will be tested on various edge devices (For now is using NVIDIA Jetson Orin) on pytorch framework.

Full library list is on `requirement.txt` file (WIP)

## Teacher
1. Fetch the pretrained teacher models (WIP) by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models` or trained on your own by running `train_teacher.py`

2. Run distillation by following commands in `scripts/run_cifar_distill.sh` (WIP). An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.

## Pruning
For now, pruning is done by changing `model_name` variabel on `prune.py` coresponding to teacher model manualy.

## Benchmarking
For benchmark, run `lat-acc-check-cifar100.py`. An example running command are as shown

    
    python3 lat-acc-check-difar100.py --path ./save/model/vgg8_last.pth --print_freq 10

The flag are explained as:
    
- `--path` : the model path that wanted to benchmark
- `--print_freq` : print frequency, default `10`

This will benchmark:

- Test Accuracy(WIP)
- Model total inferance time
- Model latency per image
    