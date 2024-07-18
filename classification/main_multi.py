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
    parser.add_argument('--doAMP', action='store_true', default=True, help='as the name says => shorter training')
    parser.add_argument('--nbblr', default=0.0, type=float, help='learning rate for maskers')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma factor for lr decay')
    parser.add_argument('--hw', default=224, type=int, help='resolution of images')
    parser.add_argument('--wandb', action='store_true')
    
    # Multi-target training
    parser.add_argument('--budget_targets', nargs='+', type=float, help='provide target budgets')
    parser.add_argument('--roundrobin', action='store_true', default=True, help='round robin schedule for training')

    # Target awareness in training 
    parser.add_argument('--lossw', default=1, type=float, help='lossw')
    parser.add_argument('--awareness_metric', type=str, default='l2', help='l1, l2, none')
    parser.add_argument('--performance_targets_csv', type=str, default='train-targets-imagenet1k.csv', help='')

    # KD
    parser.add_argument('--doKD', action='store_true', default=True, help='do KD')
    parser.add_argument('--teacher_model', type=str, default='resnet50', help='teacher network model name')
    parser.add_argument('--T_kd', default=20, type=float, help='KD temperature')

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
        project="im-shortcut-" + args.model + "-b" + "-".join(str(ee) for ee in target_densities) )
        wandb.config.update(args)

    if args.doKD:
        teacher_models = utils.loadteachers( models, args.teacher_model, device)
        print ( "Loaded ", len(teacher_models) , " teacher models")

    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=True, pretrained=args.pretrained, target_densities=target_densities).to(device=device)

    ## CRITERION
    class Loss(nn.Module):
        def __init__(self, target_density):
            super(Loss, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterionMulti(target_density, args.epochs) if target_density >= 0 else None

        def forward(self, output, target, meta):
            l = self.task_loss(output, target) 
            logger.add(str(meta['target_density'])+"/"+'loss_task', l.item())
            if self.sparsity_loss is not None:
                l += 10*self.sparsity_loss(meta)
            return l
    
    
    class LossplusKD(nn.Module):
        def __init__(self, target_density):
            super(LossplusKD, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
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
            criterions[str(target_density)] = LossplusKD(target_density)
        else:
            criterions[str(target_density)] = Loss(target_density)

        val_criterions[str(target_density)] = Loss(target_density)

    if args.nbblr != 0:
        params_dict = dict(model.named_parameters())
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
                        "weight_decay": args.weight_decay * 1, 
                    }
                )

        ## OPTIMIZER
        optimizer = torch.optim.SGD(optim_params)
        
    else:
        ## OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0
    currentprec1s = {}
    
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

    try:
        
        final_epochs = args.epochs
        adjusted_decay = args.lr_decay

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=adjusted_decay, last_epoch=start_epoch, gamma=args.gamma)
    except:
        print('Warning: Could not reload learning rate scheduler')
    start_epoch += 1

    print("* Number of trainable parameters:", utils.count_parameters(model))

    ## EVALUATION
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
        
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

    # modelparams = {}
    # for name, param in model.named_parameters():
    #     # print(name, ": ", param.requires_grad)
    #     # modelparams[name].append(param)
    #     modelparams[name] = param
    ## TRAINING
    ################################################## PHASE 1 ##################################################
    bestprec1s = {}

    for target_density in target_densities:
        bestprec1s[target_density] = 0

    for epoch in range(start_epoch, final_epochs):
        print(f"########## Wcnn + N*Wm training: Epoch {epoch} ##########")


        if epoch == start_epoch: # The first epoch it evals

            for target_density in target_densities:
                    # monitor_filename = monitorfilename[:-4] + str(target_density) + monitorfilename[-4:]
                    monitorfilename = f"{args.save_dir}/initial_t{target_density}_{formatted_datetime}.txt"
                    prec1 = validate_for_budget(args, val_loader, model, val_criterions, epoch, monitorfilename, target_density)
                    currentprec1s[target_density] = prec1

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        
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

            normalized_weights = [ratio / total*3 for ratio in ratios]

            print("Original Numbers:", ratios)
            print("Normalized Weights:", normalized_weights)
            loss_weights = normalized_weights            

        print("Final Loss weights:", loss_weights)

        if args.doKD and args.doAMP and args.roundrobin:
            train(args, train_loader, model, criterions, optimizer, epoch, target_densities, loss_weights, teacher_models)

        lr_scheduler.step()

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

        headers = []
        csvrow = []
        for num, target_density in enumerate( target_densities ):
            headers.append("prec1_" + str(target_density))
            headers.append("lossw_" + str(target_density) )
            csvrow.append( currentprec1s[target_density])
            csvrow.append( loss_weights[num])

        csv_filename = f"{args.save_dir}/lossw_monitor{formatted_datetime}.csv"
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if file.tell() == 0:
                writer.writerow(headers)
            
            writer.writerow(csvrow)

        print(f" * Best prec1: {bestprec1s}")

        dict_to_log = {}

        for num, target_density in enumerate( target_densities ):
            dict_to_log["prec1_" + str(target_density)] = currentprec1s[target_density]
            dict_to_log["lossw_" + str(target_density)] = loss_weights[num]

        dict_to_log["lr"] = optimizer.param_groups[0]['lr']

        if args.wandb:  wandb.log(dict_to_log)

    if args.wandb: wandb.finish()


def train(args, train_loader, model, criterions, optimizer, epoch, target_densities=None, loss_weights=[0.33, 0.33, 0.33], teacher_models=[]):
    print("Number of teachers: ", len(teacher_models))
    if len(teacher_models)==0:  
        print("Need KD but no teachers found. Exiting...")
        exit()
    # if utils.checkifallarethesame(teacher_models):
    if len(teacher_models)==1:
        single_teacher=True
        print( "Single KD teacher")
    else:
        single_teacher=False
        print( "Multiple KD teachers")

    for tmodel in teacher_models:
        tmodel.eval()
    """
    Run one train epoch
    """
    model.train()

    print( "EPOCH with loss_weights: ", loss_weights)

    # print(modelparams)
    # exit()
    # set gumbel temp
    # disable gumbel noise in finetuning stage
    if epoch < args.lr_decay[0]:
        gumbel_temp = 5.0
    elif epoch < args.lr_decay[1]:
        gumbel_temp = 2.5
    else:
        gumbel_temp = 1
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    counter = 0
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):


            if single_teacher is True:
                with torch.no_grad():
                    meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': False, 'epoch': epoch, 'target_density': -1}            
                    teacher_output, teacher_meta = teacher_models[0](input, meta)


            # metalist = []
            loss = 0
            # for target_index, (target_density, lossw) in enumerate(zip(target_densities, loss_weights)):

            # print ( counter, " density: ", target_densities[counter % len(target_densities)])
            # print ( counter, " lossw: ", loss_weights[counter % len(target_densities)])
            # print ( counter, " target_index: ", counter % len(target_densities))
            target_density = target_densities[counter % len(target_densities)]
            lossw = loss_weights[counter % len(target_densities)]
            target_index = counter % len(target_densities)

            # print( "Running for ", target_density)
            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch, 'target_density': target_density}
            output, meta = model(input, meta)

            if single_teacher is False:
                with torch.no_grad():
                    tmeta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': False, 'epoch': epoch, 'target_density': -1}
                    teacher_output, teacher_meta = teacher_models[target_index](input, tmeta)

            # metalist.append(meta)
            # if args.lossw == 1:
            #     lossfactor = 1
            # else:
            #     lossfactor = args.lossw 
            lossfactor = lossw
            # loss_one = lossfactor * criterions[str(meta['target_density'])](output, target, meta)            # print( "Running for ", target_density, " Loss: ", loss_one.item())
            loss_one = lossfactor *criterions[str(meta['target_density'])](output, target, meta, teacher_output)

            logger.add(str(meta['target_density'])+"/"+'target_density_loss', loss_one.item())

            loss += loss_one


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        logger.tick()        

        counter+=1

def validate_for_budget(args, val_loader, model, criterions, epoch, monitorfilename=None, target_density=None):
    """
    Run evaluation
    """

    # monitor 
    print ( monitorfilename)

    # print(model.conv1.weight[0,0,:2])
    # print(model.fc.weight[0,:5])

    top1 = utils.AverageMeter()

    # switch to evaluate mode
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

                plt.show()
                break


            # exit()


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
