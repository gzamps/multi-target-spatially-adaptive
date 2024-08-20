import argparse
import os.path

import matplotlib.pyplot as plt

import dataloader.imagenet
import dynconv
import torch
import models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import utils.flopscounter as flopscounter
import utils.logger as logger
import utils.utils as utils
import utils.viz as viz
from torch.backends import cudnn as cudnn

import torchvision

# mine
import datetime
import csv

# EMA
from timm.utils import ModelEma
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
import math
from timm.utils import get_state_dict
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import wandb


scaler = torch.cuda.amp.GradScaler() # AMP

cudnn.benchmark = True
device='cuda'

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')

    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=120, type=int, help='number of epochs')
    parser.add_argument('--model', type=str, default='resnet101', help='network model name')
    parser.add_argument('--budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--data', default='/home/gzampokas/data/imagenet', type=str, metavar='PATH',
                    help='ImageNet dataset root')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    parser.add_argument('--pretrained', action='store_true', default=True, help='start from pretrained model')
    parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    parser.add_argument('--doAMP', action='store_true', default=True, help='as the name says => shorter training')
    parser.add_argument('--nbblr', default=0.0, type=float, help='learning rate for maskers')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma factor for lr decay')
    parser.add_argument('--hw', default=224, type=int, help='resolution of images')
    
   # Multi-target training
    parser.add_argument('--budget_targets', nargs='+', type=float, help='provide target budgets')
    parser.add_argument('--roundrobin', action='store_true', default=True, help='round robin schedule for training')

    # Target awareness in training 
    parser.add_argument('--lossw', default=1, type=float, help='lossw')
    parser.add_argument('--awareness_metric', type=str, default='l2', help='l1, l2, none')
    parser.add_argument('--performance_targets_csv', type=str, default='train-targets-imagenet1k.csv', help='')

    # KD
    parser.add_argument('--doKD', action='store_true', default=True, help='do KD')
    parser.add_argument('--teacher_model', type=str, default='convnext_tiny', help='teacher network model name')
    parser.add_argument('--T_kd', default=20, type=float, help='KD temperature')

    # ConNeXt specific
    parser.add_argument('--update_freq', default=32, type=int,
                        help='gradient accumulation steps: update_freq 4, se 8 GPUs,  (in 1xGPU i need 8*4=32)') # mine

    parser.add_argument('--lr_scale', type=float, default=0.2)
    parser.add_argument('--base_rate', type=float, default='0.9')
    parser.add_argument('--ratio_weight', type=float, default='2.0') 

    # EMA related parameters
    parser.add_argument('--model_ema', type=utils.str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=utils.str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=utils.str2bool, default=True, help='Using ema to eval during training.')
    parser.add_argument('--resume_ema', default='', type=str, metavar='PATH')


    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=utils.str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)') # 0.2 tiny, 0.2 small, 0.5 base

    parser.add_argument('--wandb', action='store_true')

    args =  parser.parse_args()
    print('Args:', args)

    valcrophw = int(args.hw * 256/224) 

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.hw),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(valcrophw),
            transforms.CenterCrop(args.hw),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    # Initializing scheme
    target_densities = args.budget_targets
    print ( "Loading performance targets for ImageNet")
    training_targets = utils.get_training_performance_targets( args.budget_targets, args.model, args.performance_targets_csv)
    print ( "target_densities for ", args.model, " :", target_densities)    
    print ( "training_targets for ", args.model, " :", training_targets)

    if args.wandb:
        wandb.init(
        project="im-baseline-" + args.model + "-b" + "-".join(str(ee) for ee in target_densities) )
        wandb.config.update(args)


    teacher_models = []
    if args.doKD:
        teacher_models = utils.loadteachers( models, args.teacher_model, device)
        print ( "Loaded ", len(teacher_models) , " teacher models")

    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=True, pretrained=args.pretrained, target_densities=target_densities, drop_path_rate=args.drop_path).to(device=device)

    # print( model)
    # exit()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        task_criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        task_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        task_criterion = torch.nn.CrossEntropyLoss()


    ## CRITERION
    class Loss(nn.Module):
        def __init__(self,  target_density, task_criterion=torch.nn.CrossEntropyLoss()):
            super(Loss, self).__init__()
            # self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.task_loss = task_criterion.to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterionMulti(target_density, args.epochs) if target_density >= 0 else None

        def forward(self, output, target, meta):
            l = self.task_loss(output, target) 
            logger.add(str(meta['target_density'])+"/"+'loss_task', l.item())
            if self.sparsity_loss is not None:
                l += 10*self.sparsity_loss(meta)
            return l
    
    class LossplusKD(nn.Module):
        def __init__(self, target_density, task_criterion=torch.nn.CrossEntropyLoss()):
            super(LossplusKD, self).__init__()
            # self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.task_loss = task_criterion.to(device=device)            
            self.sparsity_loss = dynconv.SparsityCriterion(target_density, args.epochs) if target_density >= 0 else None

            self.T_kd =  args.T_kd# 5
            self.alpha_kd = 0.01 


        def forward(self, output, target, meta, teacher_output):
            l = self.task_loss(output, target) 
            # logger.add('loss_task', l.item())
            logger.add(str(meta['target_density'])+"/"+'loss_task', l.item())


            kd_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(output/self.T_kd, dim=1),
                    torch.nn.functional.softmax(teacher_output.detach()/self.T_kd, dim=1),
                    reduction='batchmean'
                ) * self.T_kd**2
            # logger.add('loss_kd', kd_loss.item())
            logger.add(str(meta['target_density'])+"/"+'loss_kd', kd_loss.item())


            if self.sparsity_loss is not None:
                l += 10*self.sparsity_loss(meta)

            l += self.alpha_kd * kd_loss

            return l

    criterions = {}
    val_criterions = {}
    for target_density in target_densities:
        if args.doKD :
            criterions[str(target_density)] = LossplusKD(target_density, task_criterion)
        else:
            criterions[str(target_density)] = Loss(target_density, task_criterion)

        val_criterions[str(target_density)] = Loss(target_density, task_criterion)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('Number of params:', n_parameters)

    total_batch_size = args.batchsize * args.update_freq *  1 #utils.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequency = %d" % args.update_freq)
    print("Number of training examples = %d" % len(train_dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)


    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    optimizer = utils.create_optimizer(
        args, model, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None,
        bone_lr_scale=args.lr_scale)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # print ( lr_schedule_values)
    # print ( len(lr_schedule_values))
    # plt.plot(range(len(lr_schedule_values)), lr_schedule_values)
    # plt.show()
    # exit()
    max_accuracy, max_accuracy_ema = 0,0

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0
    best_prec1_ema = 0
    currentprec1s = {}
    currentprec1s_ema = {}
    bestprec1s = {}
    bestprec1s_ema = {}
    for target_density in target_densities:
        currentprec1s[target_density] = 0
        currentprec1s_ema[target_density] = 0
        bestprec1s[target_density] = 0
        bestprec1s_ema[target_density] = 0

    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            for target_density in target_densities:
                currentprec1s[target_density] = checkpoint[target_density]            
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"=> loaded checkpoint '{args.resume}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)

    if args.resume_ema:
        if os.path.isfile(args.resume):
            print(f"=> loading EMA checkpoint '{args.resume_ema}'")
            checkpoint = torch.load(args.resume)
            best_prec1_ema = checkpoint['best_prec1']
            model_ema.ema.load_state_dict(checkpoint['state_dict'])
            for target_density in target_densities:
                currentprec1s_ema[target_density] = checkpoint[target_density]            
        else:
            msg = "=> no EMA checkpoint found at '{}'".format(args.resume_ema)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)


    start_epoch += 1
            
    print("* Number of trainable parameters:", utils.count_parameters(model))

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")

    ## EVALUATION
    if args.evaluate:

        ft_bestprec1s = {}
        pre_ft_bestprec1s = {}

        for target_density in target_densities:
            ft_bestprec1s[target_density] = 0


        print(f"########## multi architecture evaluation ##########")
        for target_density in target_densities:
            resume_path_str = str(args.resume).replace(".", "")
            if not os.path.exists(os.path.join(resume_path_str)):
                os.makedirs(os.path.join(resume_path_str))

            monitorfilename = f"{resume_path_str}/preft_val_t{target_density}_{formatted_datetime}.txt"            
            prec1 = validate_for_budget(args, val_loader, model, val_criterions, start_epoch, monitorfilename, target_density=target_density)
            ft_bestprec1s[target_density] = prec1
            pre_ft_bestprec1s[target_density] = prec1

        print(f" * multi architecture best prec1: {pre_ft_bestprec1s}")
        return
        
    ## TRAINING
    print(f"########## TRAINING ##########")
    print ( "Trainig maskers with densities ", target_densities)
    print ( "[\u2713] Target Aware Training")
    print ( "Targets: ", training_targets)
    print ( "Norm type: ", args.awareness_metric)
   
    if args.doKD:
        print ( "[\u2713] Using Knowledge Distillation")
        print ( "Teachers: ", args.teacher_model)
    else:
        print ( "[ ] No Knowledge Distillation")

    print(f"########## ######## ##########")

    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")

        if epoch == start_epoch: # The first epoch it evals
            print(f"########## Evaluation ##########")

            for target_density in target_densities:
                    # monitor_filename = monitorfilename[:-4] + str(target_density) + monitorfilename[-4:]
                    monitorfilename = f"{args.save_dir}/initial_t{target_density}_{formatted_datetime}.txt"
                    prec1 = validate_for_budget(args, val_loader, model, val_criterions, epoch, monitorfilename, target_density)
                    currentprec1s[target_density] = prec1
            
        if epoch == 0 or args.awareness_metric == 'none':
            # loss_weights = [0.33, 0.33, 0.33]
            loss_weights = [1.0] * len(target_densities)
        else:
            currentprec1s_ = []
            for target_density in target_densities:
                currentprec1s_.append( currentprec1s[target_density])

            if args.awareness_metric == 'l1':
                ratios = [ 1 - curr/target for curr,target in zip(currentprec1s_,training_targets) ] 
            elif args.awareness_metric == 'l2':
                ratios = [ (target - curr)*(target - curr) for curr,target in zip(currentprec1s_,training_targets) ] 
            else:
                print("Specify correct args.awareness_metric")

            total = sum(ratios)

            normalized_weights = [ratio / total*len(target_densities) for ratio in ratios]

            print("Original Numbers:", ratios)
            print("Normalized Weights:", normalized_weights)
            loss_weights = normalized_weights

        print("Final Loss weights:", loss_weights)

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(args, train_loader, model, criterions, optimizer, epoch,
                        device, loss_scaler,
                        target_densities = target_densities, # shortcut
                        loss_weights = loss_weights, # shortcut
                        teacher_models = teacher_models, # shortcut
                        max_norm=args.clip_grad, 
                        model_ema=model_ema, 
                        mixup_fn=mixup_fn,
                        # log_writer=log_writer,
                        start_steps=epoch * num_training_steps_per_epoch,
                        lr_schedule_values=lr_schedule_values,
                        wd_schedule_values=wd_schedule_values,
                        num_training_steps_per_epoch=num_training_steps_per_epoch, 
                        update_freq=args.update_freq,
                        use_amp=args.doAMP)

        # evaluate on validation set
        for target_density in target_densities:
            # monitor_filename = monitorfilename[:-4] + str(target_density) + monitorfilename[-4:]
            monitorfilename = f"{args.save_dir}/train_t{target_density}_{formatted_datetime}.txt"
            prec1 = validate_for_budget(args, val_loader, model, val_criterions, epoch, monitorfilename, target_density)
            currentprec1s[target_density] = prec1

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        ###### SAVING CRITERION
        is_best = utils.update_model_criterion( bestprec1s, currentprec1s, target_densities)

        save_checkpoint_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
        }
        for target_density in target_densities:
            save_checkpoint_dict[target_density] = bestprec1s[target_density]

        utils.save_checkpoint(save_checkpoint_dict, folder=args.save_dir, is_best=is_best)

        if is_best is True:
            acc_paretto_plot_filename = f"{args.save_dir}/prec1_paretto_{formatted_datetime}.png"                
            utils.save_best_accuracy_paretto_while_training(acc_paretto_plot_filename,  target_densities, bestprec1s, training_targets)

       # evaluate on validation set EMA
        for target_density in target_densities:
            # monitor_filename = monitorfilename[:-4] + str(target_density) + monitorfilename[-4:]
            monitorfilename_ema = f"{args.save_dir}/train_ema_t{target_density}_{formatted_datetime}.txt"
            prec1_ema = validate_for_budget(args, val_loader, model_ema.ema, val_criterions, epoch, monitorfilename_ema, target_density)
            currentprec1s_ema[target_density] = prec1_ema

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        best_prec1_ema = max(prec1_ema, best_prec1_ema)

        ###### SAVING CRITERION
        is_best_ema = utils.update_model_criterion( bestprec1s_ema, currentprec1s_ema, target_densities)

        save_checkpoint_dict_ema = {
            'state_dict': get_state_dict(model_ema),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1_ema,
        }
        for target_density in target_densities:
            save_checkpoint_dict_ema[target_density] = bestprec1s_ema[target_density]

        utils.save_checkpoint_ema(save_checkpoint_dict_ema, folder=args.save_dir, is_best=is_best_ema)

        if is_best_ema is True:
            acc_paretto_plot_filename = f"{args.save_dir}/prec1_paretto_ema_{formatted_datetime}.png"                
            utils.save_best_accuracy_paretto_while_training(acc_paretto_plot_filename,  target_densities, bestprec1s_ema, training_targets)

        dict_to_log = {}

        for target_density in target_densities:
            dict_to_log["prec1_" + str(target_density)] = currentprec1s[target_density]
            dict_to_log["prec1_ema" + str(target_density)] = currentprec1s_ema[target_density]

        dict_to_log["lr"] = optimizer.param_groups[0]['lr']


        if args.wandb:  wandb.log(dict_to_log)

    if args.wandb: wandb.finish()


def train(args, train_loader, model, criterions, optimizer, epoch,
                    device, loss_scaler, 
                    target_densities = [0.3, 0.5, 0.9], loss_weights = [1.0, 1.0, 1.0], teacher_models = [], # shortcut
                    max_norm = 0,
                    model_ema = None, mixup_fn = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    print( "----- train_one_epoch ------ epoch = ", epoch)


    print("Number of teachers: ", len(teacher_models))
    if len(teacher_models)==0:  
        print("Need KD but no teachers found. Exiting...")
        # exit()
    # if utils.checkifallarethesame(teacher_models):
    if len(teacher_models)==1:
        single_teacher=True
        print( "Single KD teacher")
    else:
        single_teacher=False
        print( "Multiple KD teachers")

    for tmodel in teacher_models:
        tmodel.eval()

    print( "EPOCH with loss_weights: ", loss_weights)

    model.train(True)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    num_step =  len(train_loader)

    if epoch <  0.3*args.epochs:
        gumbel_temp = 5.0
    elif epoch <  0.6*args.epochs:
        gumbel_temp = 2.5
    else:
        gumbel_temp = 1
    gumbel_noise = False if epoch > 0.8*args.epochs else True


    optimizer.zero_grad()
    data_iter_step = 0
    counter = 0    
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        # round robin index        
        target_density = target_densities[counter % len(target_densities)]
        lossw = loss_weights[counter % len(target_densities)]
        target_index = counter % len(target_densities)
        # print("TRAINING FOR :", target_density)
        loss = 0

        if use_amp:
            with torch.cuda.amp.autocast():
                meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch, 'target_density': target_density}
                output, meta = model(input, meta)

                lossfactor = lossw

                # run teachers                
                if len(teacher_models)==0:
                    loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta)
                elif len(teacher_models)==1:
                    with torch.no_grad():
                        tmeta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': False, 'epoch': epoch, 'target_density': -1}            
                        teacher_output, teacher_meta = teacher_models[0](input, tmeta)
                    loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)
                else:
                    with torch.no_grad():
                        tmeta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': False, 'epoch': epoch, 'target_density': -1}
                        teacher_output, teacher_meta = teacher_models[target_index](input, tmeta)
                    loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)

                # loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)
                logger.add(str(meta['target_density'])+"/"+'target_density_loss', loss_one.item())

                loss += loss_one

        else: # full precision
            meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch, 'target_density': target_density}
            output, meta = model(input, meta)

            lossfactor = lossw
            # loss_one = lossfactor * criterions[str(meta['target_density'])](output, target, meta)            # print( "Running for ", target_density, " Loss: ", loss_one.item())

            # run teachers                
            if len(teacher_models)==0:
                loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta)
            elif len(teacher_models)==1:
                with torch.no_grad():
                    tmeta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': False, 'epoch': epoch, 'target_density': -1}            
                    teacher_output, teacher_meta = teacher_models[0](input, tmeta)
                loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)
            else:
                with torch.no_grad():
                    tmeta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': False, 'epoch': epoch, 'target_density': -1}
                    teacher_output, teacher_meta = teacher_models[target_index](input, tmeta)
                loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)

            # loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)
            logger.add(str(meta['target_density'])+"/"+'target_density_loss', loss_one.item())

            loss += loss_one

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == target).float().mean()
        else:
            class_acc = None

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]

        data_iter_step += 1
        counter += 1
        

def validate_for_budget(args, val_loader, model, criterions, epoch, monitorfilename=None, target_density=None):
    """
    Run evaluation
    """

    # monitor 
    print ( monitorfilename)

    # print(model.conv1.weight[0,0,:2])
    # print(model.fc.weight[0,:5])

    top1 = utils.AverageMeter()

    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch, 'target_density': target_density}
            output, meta = model(input, meta)
            output = output.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

            # if args.plot_ponder:
            if args.plot_ponder:

                viz.plot_image(input)
                viz.plot_ponder_cost(meta['masks'])
                viz.plot_masks(meta['masks'])
                viz.plot_mask_distributions(meta['masks'])
                viz.plot_mask_soft(meta['masks'])
                # print( target_density )
                plt.show()
                break

    print(f'* Target_density {target_density} ')
    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f}')
    mmacs = model.compute_average_flops_cost()[0]/1e6
    # print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {mmacs:.6f} MMac')
    values = [epoch, mmacs, top1.avg]

    if monitorfilename is not None:
        with open(monitorfilename, 'a') as file:
            file.write(','.join(map(str, values)) + '\n')
            utils.save_accuracy_while_training(monitorfilename)

    model.stop_flops_count()


    return top1.avg

if __name__ == "__main__":
    main()    