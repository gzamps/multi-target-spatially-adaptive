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
device='cpu'

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')
    parser.add_argument('--lr_decay', default=[30,60,90], nargs='+', type=int, help='learning rate decay epochs')
    parser.add_argument('--batchsize', default=64, type=int, help='batch size')
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
    parser.add_argument('--doAMP', action='store_true', help='as the name says => shorter training')
    parser.add_argument('--nbblr', default=0.0, type=float, help='learning rate for maskers')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma factor for lr decay')
    parser.add_argument('--hw', default=224, type=int, help='resolution of images')


    # DEPDNDS
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    parser.add_argument('--lr_scale', type=float, default=0.2)
    parser.add_argument('--base_rate', type=float, default='0.9')
    parser.add_argument('--ratio_weight', type=float, default='2.0')

    # EMA related parameters
    parser.add_argument('--model_ema', type=utils.str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=utils.str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=utils.str2bool, default=True, help='Using ema to eval during training.')


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

    if args.wandb:
        wandb.init(
        project="im-baseline-" + args.model + "-b" + str(args.budget) )
        wandb.config.update(args)

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


    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0, pretrained=args.pretrained, drop_path_rate=args.drop_path).to(device=device)

    # Measure FLOPs of base/dense model
    if args.budget == -1:
        from flopth import flopth
        flops, params = flopth(model, in_size=((3, 224, 224),), show_detail=True)
        print ( flops)
        print (params)
        exit()

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
        def __init__(self,  task_criterion=torch.nn.CrossEntropyLoss()):
            super(Loss, self).__init__()
            # self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.task_loss = task_criterion.to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, args.epochs) if args.budget >= 0 else None

        def forward(self, output, target, meta):
            l = self.task_loss(output, target) 
            logger.add('loss_task', l.item())
            if self.sparsity_loss is not None:
                l += 10*self.sparsity_loss(meta)
            return l
    
    criterion = Loss(task_criterion)

    print(args.model)

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

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batchsize * args.update_freq *  1 #utils.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
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
    # plt.figure(1)
    # plt.plot(range(len(lr_schedule_values)), lr_schedule_values)
    # plt.figure(2)
    # plt.plot(range(len(wd_schedule_values)), wd_schedule_values)

    # plt.show()
    max_accuracy, max_accuracy_ema = 0,0

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0
    best_prec1_ema = 0

    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"=> loaded checkpoint '{args.resume}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)

    start_epoch += 1
            
    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")


    ## EVALUATION
    if args.evaluate:
        print(f"########## Evaluation ##########")
        monitorfilename = f"{args.resume}/evaluate_t{args.budget}_{formatted_datetime}.txt"            
        prec1 = validate(args, val_loader, model, criterion, start_epoch, monitorfilename)
        wandb.finish()
        return
        
    ## TRAINING
    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")

        if epoch == start_epoch: # The first epoch it evals
            print(f"########## Evaluation ##########")
            monitorfilename = f"{args.save_dir}/preft_val_t{args.budget}_{formatted_datetime}.txt"            
            prec1 = validate(args, val_loader, model, criterion, start_epoch, monitorfilename)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train_one_epoch(args, train_loader, model, criterion, optimizer, epoch,
                        device, loss_scaler,
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


        monitorfilename = f"{args.save_dir}/train_t{args.budget}_{formatted_datetime}.txt"
        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, epoch, monitorfilename)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
        }, folder=args.save_dir, is_best=is_best)

        print(f" * Best prec1: {best_prec1}")

        monitorfilename_ema = f"{args.save_dir}/train_ema_t{args.budget}_{formatted_datetime}.txt"

        prec1_ema = validate(args, val_loader, model_ema.ema, criterion, epoch, monitorfilename_ema)

        # remember best prec@1 and save checkpoint
        is_best = prec1_ema > best_prec1_ema
        best_prec1_ema = max(prec1_ema, best_prec1_ema)
        utils.save_checkpoint_ema({
            'state_dict': get_state_dict(model_ema), #.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1_ema,
        }, folder=args.save_dir, is_best=is_best)

        print(f" * Best prec1: {best_prec1}")

        if args.wandb:  wandb.log({"prec1": prec1, "prec1_ema": prec1_ema,
            "lr": optimizer.param_groups[0]['lr'], })

    if args.wandb: wandb.finish()


def validate(args, val_loader, model, criterion, epoch, monitorfilename):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()

    model = flopscounter.add_flops_counting_methods(model) 
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    data_iter_step = 0
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
            output, meta = model(input, meta)
            output = output.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

            if args.plot_ponder:
                viz.plot_image(input)
                viz.plot_ponder_cost(meta['masks'])
                viz.plot_masks(meta['masks'])
                viz.showKey()

            # break
            data_iter_step+=1

    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f}')
    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    mmacs = model.compute_average_flops_cost()[0]/1e6

    values = [epoch, mmacs, top1.avg]
    if monitorfilename is not None:
        with open(monitorfilename, 'a') as file:
            file.write(','.join(map(str, values)) + '\n')
            utils.save_accuracy_while_training(monitorfilename)


    model.stop_flops_count()
    return top1.avg


def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch,
                    device, loss_scaler, max_norm = 0,
                    model_ema = None, mixup_fn = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    print( "Epoch = ", epoch)

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

        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}

        if use_amp:
            with torch.cuda.amp.autocast():
                # output = model(samples)
                output, meta = model(input, meta)
                loss = criterion(output, target, meta)
        else: 
            output, meta = model(input, meta)
            loss = criterion(output, target, meta)


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

if __name__ == "__main__":
    main()    