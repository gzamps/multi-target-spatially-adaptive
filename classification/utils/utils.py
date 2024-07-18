import os.path

import torch
from torchvision import transforms

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import json
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import math 
from torch._six import inf

import models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

    "convnext_tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    # "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    # "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    # "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    # "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    # "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, folder, is_best, is_ft=False):
    """
    Save the training model
    """
    if len(folder) == 0:
        print('Did not save model since no save directory specified in args!')
        return
        
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, 'checkpoint.pth')
    print(f" => Saving {filename}")
    torch.save(state, filename)

    if is_best:
        filename = os.path.join(folder, 'checkpoint_best.pth')
        print(f" => Saving best {filename}")
        torch.save(state, filename)
    
    if is_ft:
        filename = os.path.join(folder, 'checkpoint_ft.pth')
        print(f" => Saving FT  {filename}")
        torch.save(state, filename)


def save_checkpoint_ema(state, folder, is_best):
    """
    Save the training model
    """
    if len(folder) == 0:
        print('Did not save model since no save directory specified in args!')
        return
        
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, 'checkpoint_ema.pth')
    print(f" => Saving {filename}")
    torch.save(state, filename)

    if is_best:
        filename = os.path.join(folder, 'checkpoint_best_ema.pth')
        print(f" => Saving best {filename}")
        torch.save(state, filename)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        assert tensor.shape[0] == 3
        return self.unnormalize(tensor)


def update_model_criterion( bestprec1s, currentprec1s, target_densities ):
    # is_best = False

    # all_best = True
    # for target_density in target_densities:
    #     if currentprec1s[target_density] < bestprec1s[target_density]:
    #         all_best = False
    #         break

    # # precs best for all densities
    # if all_best: return True
    diff = 0
    for target_density in target_densities:
        diff += currentprec1s[target_density] - bestprec1s[target_density]
        print( "diff at " , target_density, " = ", currentprec1s[target_density] - bestprec1s[target_density])

    print("total diff = ", diff)
    if diff > 0:
        for target_density in target_densities:
            bestprec1s[target_density] = max( bestprec1s[target_density], currentprec1s[target_density])
        return True
    else:
        return False


def get_training_performance_targets( budget_targets, model_name, performance_targets_csv):

    budget_targets_to_str = [str(training_target) for training_target in budget_targets]
    df = pd.read_csv(performance_targets_csv, index_col='model')
    print ( df)

    if model_name.endswith("_multi"):
        model_name=model_name.rstrip("_multi")

    performance_targets = df.loc[model_name, budget_targets_to_str].tolist()

    return performance_targets


def set_bn_eval_mode(module):
    """
    Recursively sets all BatchNorm layers to evaluation mode.
    """
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()
    for child_module in module.children():
        set_bn_eval_mode(child_module)


def save_accuracy_while_training(monitorfilename):

    x_values = []
    y_values = []

    filename = monitorfilename

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines):
            # Split the line by commas and convert the third value to a float
            values = line.strip().split(',')
            if len(values) >= 3:
                x_values.append(line_number)
                y_values.append(float(values[2]))

    if len(x_values) == 0 or len(y_values) ==0:
        return

    max_index = np.argmax(y_values)
    max_x = x_values[max_index]
    max_y = y_values[max_index]

    plt.figure(figsize=(12, 6))
    # Create a plot
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Data')
    plt.xlabel('Epoch')
    plt.ylabel('Prec@1')
    plt.title('Accuracy evolution over training epochs')


    # Annotate the maximum value with a red arrow
    plt.annotate(f'Max: {max_y} [ep:{max_index}]', xy=(max_x, max_y), xytext=(max_x, max_y + 5),
                 arrowprops=dict(arrowstyle='->', color='red'), color='red')

    # plt.legend()
    plt.savefig( filename[:-4] + ".png", dpi=400) #, transparent=True)
    plt.close('all')
    # plt.show()


def save_best_accuracy_paretto_while_training(monitorfilename, x_labels, y1_data, y2_data):

    y1_data = list(y1_data.values())

    plt.figure(1)

    if len(x_labels) != len(y1_data) or len(x_labels) != len(y2_data):
        raise ValueError("Input lists must have the same length.")
    plt.plot(x_labels, y1_data, marker='o', linestyle='-', color='red', label='Target accuracies')

    plt.plot(x_labels, y2_data, marker='o', linestyle='-', color='b', label='Current accuracies')
    plt.xlabel('Sparsity targets ') #, x_labels )
    plt.ylabel('Prec@1')
    plt.title('Best Prec@1 evolution over training epochs')

    plt.savefig( monitorfilename, dpi=400) #, transparent=True)

    # plt.close('all')
    plt.close()        
    plt.clf()


def convert_best_ckpts_to_one(folder, target_densities):
    
    print( "Finetune done! Combining multiple ckpts to one.")

    for target_density in target_densities:

        filename = os.path.join(folder, 'checkpoint_best-ft-d' + str(target_density).replace('.', '') + '.pth')
        print(f" => Loading {filename}")

    return


def loadteachers( models, teacher_model_name, device='cuda'):

    print("Loading Teacher model(s)")
    print ("Loading single teacher")

    teacher_models = []

    teacher_net_module = models.__dict__[teacher_model_name]       
    teacher_model = teacher_net_module(sparse=False, pretrained=False).to(device=device)

    teacher_state_dict = load_state_dict_from_url(model_urls[teacher_model_name],
                                              progress=True)
    teacher_model.load_state_dict(teacher_state_dict, strict=False)

    teacher_model.eval()
    teacher_models.append(teacher_model)
    print ( "[\u2713] Succesfully Loaded teacher checkpoint of ", teacher_model_name)
    # exit()
            
    return teacher_models


def checkifallarethesame(listwithnames):
    # print( listwithnames )
    return all(i == listwithnames[0] for i in listwithnames)
    # exit()


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, bone_lr_scale=0.01):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif 'cls_token' in name or 'pos_embed' in name:
            continue # frozen weights
        elif 'fast' in name or 'predictor' in name or 'mse' in name or 'fastmlp' in name or 'masker' in name:
        # elif 'predictor' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = 'new_param_no_decay'
                this_weight_decay = 0
            else:
                group_name = 'new_param'
                this_weight_decay = weight_decay
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            if group_name is 'decay':
                scale = bone_lr_scale
                fix_step = 5
            elif group_name is 'no_decay':
                scale = bone_lr_scale
                fix_step = 5
            else:
                scale = 1.
                fix_step = 0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "fix_step": fix_step
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "fix_step": fix_step
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, bone_lr_scale=0.01):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, bone_lr_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    # parameters = model.get_costum_param_groups(args.weight_decay)

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule
