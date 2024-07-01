import argparse
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import prune

from helper.util import save_model
import CONFIG

from models import model_dict
import os

LAYER_KINDS = {
    "Linear": (nn.Linear),
    "Conv2d": (nn.Conv2d),
    "Complete": (nn.Linear, nn.Conv2d) 
}

SPARSE_LAYOUT = {
    "sparse_coo": torch.sparse_coo,
    "sparse_csr": torch.sparse_csr,
    "sparse_csc": torch.sparse_csc,
    "sparse_bsr": torch.sparse_bsr,
    "sparse_bsc": torch.sparse_bsc,
}

def prune_model(
    model: nn.Module,
    layer_kind: tuple[nn.Module],
    amount=0.5,
    n=Optional[int],
    dim=Optional[int],
):
    """
    Prune the model with the pruning method, and amount of sparsity of your choice

    Args:
        - model: The loaded model you want to prune
        - layer_kind: What layer to prune 
        - pruning_method: What pruning_method to use
        - amount (float): How much of the layer (or network) will be zeroed. 0.5 is equal to 50%
        - global_unstructured (bool): set to True to use global_unstructured pruning
        - n (int): Regularization to use (1 for L1, 2 for L2)
        - dim (int): Defines which way pruning is done (filter, input/output)
    """
    for _, module in model.named_modules():
        if isinstance(module, layer_kind):  # type: ignore
            if n is not None and dim is not None:
                prune.ln_structured(module, amount=amount, n=int(n), dim=int(dim), name="weight")  # type: ignore
            # TODO: FIX BUG for random_structured
            elif dim is not None and n is None:
                prune.random_structured(module, amount=amount, dim=int(dim), name="weight")  # type: ignore

    # Step 2. Remove the pruning mask 
    for _, module in model.named_modules():
        if isinstance(module, layer_kind):  # type: ignore
            prune.remove(module, name="weight")

    return model


def check_pruning_percentage(model: nn.Module):
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

    print(f'Total sparsity: {100 * (1 - total_nonzero_weights / total_weights):.2f}%')


def convert_to_sparse(model: nn.Module, layer_kind, sparse_layout, blocksize=None):
    for _, module in model.named_modules():
        if isinstance(module, layer_kind):
            if blocksize is not None:
                module.weight = nn.Parameter(module.weight.to_sparse(layout=sparse_layout, blocksize=blocksize))
            else:
                module.weight = nn.Parameter(module.weight.to_sparse(layout=sparse_layout))
            
    return model

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
    model_name = get_teacher_name(args.path)
    model_to_prune = model_dict[model_name[0]](num_classes=100)
    model_to_prune.load_state_dict(torch.load(args.path)['model'])
    model_to_prune.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('==> done')
    
    # model_to_prune: nn.Module = torch.load(args.path).to(CONFIG.DEVICE)
    layer_kind = LAYER_KINDS[args.layer_kind]

    pruned_model = prune_model(
        model=model_to_prune,
        layer_kind=layer_kind,
        amount=args.amount,
        n=args.n,
        dim=args.dim
    )

    if args.check_percentage:
        check_pruning_percentage(pruned_model)

    #! sparse code
    # if args.blocksize is not None:
    #     sparse_model = convert_to_sparse(
    #         model=pruned_model, 
    #         layer_kind=layer_kind, 
    #         sparse_layout=SPARSE_LAYOUT[args.sparse_layout], 
    #         blocksize=tuple(args.blocksize)
    #     )
    # else:
    #     sparse_model = convert_to_sparse(pruned_model, layer_kind, SPARSE_LAYOUT[args.sparse_layout])
    
    if not os.path.isdir(f'./save/models/{model_name[0]}'):
        os.makedirs(f'./save/models/{model_name[0]}')
    
    save_model(pruned_model, 'state-dict', f'./save/models/{model_name[0]}/{model_name[0]}_l{args.n}_{args.layer_kind}_{args.amount}_pruned.pth')

    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--layer-kind", choices=list(LAYER_KINDS.keys()), required=True)
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--n", type=int, help="Regularization to choose", choices=[1, 2])
    parser.add_argument("--dim", required=True, help="Pruning direction (channels, filters, etc..)")
    #parser.add_argument("--sparse-layout", choices=list(SPARSE_LAYOUT.keys()), required=True)
    #parser.add_argument("--save-path", required=True)
    #parser.add_argument("--blocksize", nargs="*", help="Set this for sparse_bsc and sparse_bsr", type=int)
    parser.add_argument("--check-percentage", action="store_true", default=True)

    args = parser.parse_args()
    print(args)

    main(args)
