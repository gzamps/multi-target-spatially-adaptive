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
import wandb

scaler = torch.cuda.amp.GradScaler() # AMP

cudnn.benchmark = True
device='cuda'

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[30,60,90], nargs='+', type=int, help='learning rate decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--batchsize', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
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

    parser.add_argument('--wandb', action='store_true')

    args =  parser.parse_args()
    print('Args:', args)


    if args.wandb:
        wandb.init(
        # set the wandb project where this run will be logged
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
    model = net_module(sparse=args.budget >= 0, pretrained=args.pretrained).to(device=device)


    # ################################3 FLOPS AND PARAMS ANALYSIS
    if not args.budget >= 0:
        from thop import profile
        from thop import clever_format
        input = torch.randn(1, 3, args.hw, args.hw).cuda()

        macs, params = profile(model, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.3f")
        print ( "macs: ", macs)
        print ( "params: ", params)

    ###############################3 FLOPS AND PARAMS ANALYSIS

    # # BETTER
    # if args.budget == -1:
    #     from flopth import flopth
    #     flops, params = flopth(model, in_size=((3, 224, 224),), show_detail=True)
    #     print ( flops)
    #     print (params)
    #     exit()
    ###############################3#############################

    ## CRITERION
    class Loss(nn.Module):
        def __init__(self):
            super(Loss, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, args.epochs) if args.budget >= 0 else None

        def forward(self, output, target, meta):
            l = self.task_loss(output, target) 
            logger.add('loss_task', l.item())
            if self.sparsity_loss is not None:
                l += 10*self.sparsity_loss(meta)
            return l
    
    criterion = Loss()


    if args.nbblr != 0:
        params_dict = dict(model.named_parameters())
        # print ( params_dict.items() )
        # for k, param in params_dict.items():
        #     print ( k )
        NONBACKBONE_KEYWORDS = ['masker']
        bb_lr = []
        nbb_lr = []
        nbb_keys = set()
        for k, param in params_dict.items():
            if any(part in k for part in NONBACKBONE_KEYWORDS):
                nbb_lr.append(param)
                nbb_keys.add(k)
            else:
                bb_lr.append(param)
        print ( "\nNBB KEYS: ")
        print(nbb_keys)

        optim_params = [{'params': bb_lr, 'lr': args.lr}]
        optim_params.append(
                    {
                        "params": nbb_lr,
                        "lr": args.nbblr * 1,
                        "weight_decay": args.weight_decay * 1, # 0.1  C.segblocks_policy_wd_factor,
                    }
                )

        ## OPTIMIZER
        optimizer = torch.optim.SGD(optim_params) #, args.lr,
                            # momentum=args.momentum,
                            # weight_decay=args.weight_decay)
        
    else:
        ## OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0

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


    try:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=args.lr_decay, last_epoch=start_epoch, gamma=args.gamma)
    except:
        print('Warning: Could not reload learning rate scheduler')
    start_epoch += 1
            
    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")


    ## EVALUATION
    if args.evaluate:
        # evaluate on validation set
        print(f"########## Evaluation ##########")
        monitorfilename = f"{args.resume}/evaluate_t{args.budget}_{formatted_datetime}.txt"            
        prec1 = validate(args, val_loader, model, criterion, start_epoch, monitorfilename)
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
        if args.doAMP:
            trainamp(args, train_loader, model, criterion, optimizer, epoch)
        else:
            train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()


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


        if args.wandb:  wandb.log({"prec1": prec1, 
            "lr": optimizer.param_groups[0]['lr'], })

    if args.wandb: wandb.finish()


def trainamp(args, train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    model.train()
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

    if epoch < args.lr_decay[0]:
        gumbel_temp = 5.0
    elif epoch < args.lr_decay[1]:
        gumbel_temp = 2.5
    else:
        gumbel_temp = 1
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
            output, meta = model(input, meta)
            loss = criterion(output, target, meta)

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ####################################### DO WITH AMP #######################################
        # optimizer.zero_grad()
        # with torch.cuda.amp.autocast(dtype=torch.float16):

        #     # compute output
        #     meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        #     output, meta = model(input, meta)
        #     loss = criterion(output, target, meta)

        # # Casts operations to mixed precision

        # # Scales the loss, and calls backward()
        # # to create scaled gradients
        # scaler.scale(loss).backward()

        # # Unscales gradients and calls
        # # or skips optimizer.step()
        # scaler.step(optimizer)

        # # Updates the scale for next iteration
        # scaler.update()
        ############################################################################################



        logger.tick()
        # break

def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    model.train()

    if epoch < args.lr_decay[0]:
        gumbel_temp = 5.0
    elif epoch < args.lr_decay[1]:
        gumbel_temp = 2.5
    else:
        gumbel_temp = 1
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        output, meta = model(input, meta)
        loss = criterion(output, target, meta)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        ####################################### DO WITH AMP #######################################
        # optimizer.zero_grad()
        # with torch.cuda.amp.autocast(dtype=torch.float16):

        #     # compute output
        #     meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        #     output, meta = model(input, meta)
        #     loss = criterion(output, target, meta)

        # # Casts operations to mixed precision

        # # Scales the loss, and calls backward()
        # # to create scaled gradients
        # scaler.scale(loss).backward()

        # # Unscales gradients and calls
        # # or skips optimizer.step()
        # scaler.step(optimizer)

        # # Updates the scale for next iteration
        # scaler.update()
        ############################################################################################



        logger.tick()

def validate(args, val_loader, model, criterion, epoch, monitorfilename):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model) # kanei init mia mask=None se oti einai supported Conv,linear,avgpool
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
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

    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f}')
    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    mmacs = model.compute_average_flops_cost()[0]/1e6

    # print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost_global()[0]/1e6:.6f} MMac')
    # mmacs = model.compute_average_flops_cost_global()[0]/1e6

    values = [epoch, mmacs, top1.avg]
    if monitorfilename is not None:
        with open(monitorfilename, 'a') as file:
            file.write(','.join(map(str, values)) + '\n')
            utils.save_accuracy_while_training(monitorfilename)


    model.stop_flops_count()
    return top1.avg

if __name__ == "__main__":
    main()    