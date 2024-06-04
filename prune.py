import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
from torch.nn.utils import prune
from models.vgg import vgg8, vgg11, vgg13, vgg16, vgg19
import os

model_name = 'vgg13' #diganti manual
model_path = f'save/models/vgg/{model_name}_cifar100_lr_0.05_decay_0.0005_trial_0/{model_name}_last.pth'
prune_path = f'./save/models/vgg/pruned/{model_name}_cifar100_lr_0.05_decay_0.0005_trial_0/'

if not os.path.isdir(prune_path):
    os.makedirs(prune_path)

func_lib = {'vgg11' : vgg11(),
            'vgg13' : vgg13(),
            'vgg16' : vgg16(),
            'vgg19' : vgg19()}
# Model structure
model = func_lib[model_name]
model.state_dict(torch.load(model_path))

model.to("cpu")  # Frequent crashes on GPU (GTX 950M), so set to CPU
print('model loaded!')


# Prune
# Identify the parameters to prune
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        parameters_to_prune.append((module, 'weight'))

# Apply global unstructured pruning
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,  # or prune.RandomUnstructured
    amount=0.5,  # Pruning 50% of the parameters globally
)

# Remove the reparameterization
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)


# Verify the pruning
def count_nonzero_weights(module):
    return torch.sum(module.weight != 0).item()

total_nonzero_weights = 0
total_weights = 0

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nonzero = count_nonzero_weights(module)
        total_nonzero_weights += nonzero
        total_weights += module.weight.nelement()
        print(f'{name}: {nonzero}/{module.weight.nelement()}')

print('model pruned!')
print(f'Total sparsity: {100 * (1 - total_nonzero_weights / total_weights):.2f}%')

# Save model structure and weights
torch.save(model, f'./save/models/vgg/pruned/{model_name}_cifar100_lr_0.05_decay_0.0005_trial_0/{model_name}_last.pth')

# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#         module.weight = nn.Parameter(module.weight.to_sparse())

# torch.save(model, f'./save/models/pruned/{model_name}_cifar100_lr_0.05_decay_0.0005_trial_0/{model_name}_sparse_last_pruned.pth')
