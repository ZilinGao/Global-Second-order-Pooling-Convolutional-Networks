## Created by: Zilin Gao
## Implementation of "Global Second-order Pooling Convolutional Networks", Zilin Gao, Jiangtao Xie, Qilong Wang, Peihua Li. IN CVPR2019.
## Copyright (c) 2019 Zilin Gao, Jiangtao Xie, Qilong Wang and Peihua Li.
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 

import argparse
import pdb
import os
import random
import shutil
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torchviz import  make_dot, make_dot_from_trace
from torchvision import transforms
from torchvision import datasets
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=95, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--attpos', default=None, type=list, nargs='+',
                    help='choose the position attention module inserted')
parser.add_argument('--attdim', default=128, type=int,
                    help='define the dimension of attention module')
parser.add_argument('--GSoP_mode', default=1, type=int,
                    help='GSoP-Net mode select: GSoP-Net1 or GSop-Net2')
parser.add_argument('--modeldir', default=None, type=str,
                    help='define the dimension of attention module')
best_prec1 = 0

class stats:
    def __init__(self, path, start_epoch):
        if start_epoch is not 0:
           stats_ = sio.loadmat(os.path.join(path,'stats.mat'))
           data = stats_['data']
           content = data[0,0]
           self.trainObj = content['trainObj'][:,:start_epoch].squeeze().tolist()
           self.trainTop1 = content['trainTop1'][:,:start_epoch].squeeze().tolist()
           self.trainTop5 = content['trainTop5'][:,:start_epoch].squeeze().tolist()
           self.valObj = content['valObj'][:,:start_epoch].squeeze().tolist()
           self.valTop1 = content['valTop1'][:,:start_epoch].squeeze().tolist()
           self.valTop5 = content['valTop5'][:,:start_epoch].squeeze().tolist()
        else:
           self.trainObj = []
           self.trainTop1 = []
           self.trainTop5 = []
           self.valObj = []
           self.valTop1 = []
           self.valTop5 = []

    
def main():
    pdb.set_trace()
    global args, best_prec1
    args = parser.parse_args()

    print("{} epochs total". format(args.epochs))
    if args.arch == 'resnet50':
       args.attpos = [['0']*2+['1'],['0']*3+['1'],['0']*5+['1'],['0']*2+['1']]
    elif args.arch == 'resnet101':
       args.attpos = [['0']*2+['1'],['0']*3+['1'],['0']*22+['1'],['0']*2+['1']]
    else :
       assert False, "invalid arch"

    if args.GSoP_mode == 1:
       args.epochs = 95
    elif args.GSoP_mode == 2:
       args.epochs = 65
       if args.arch == 'resnet50':
          args.attpos[-1][-1]='0'
       elif args.arch == 'resnet101':
          args.attpos[-1][-1]='0'
    else :
       assert False, "invalid GSoP mode"
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, att_position = args.attpos, 
                                           att_dim = args.attdim,spa_h = args.spa_h, GSoP_mode = args.GSoP_mode)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](att_position = args.attpos,att_dim = args.attdim,
                                           GSoP_mode = args.GSoP_mode)

    
    vizNet(model, args.modeldir)
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    # make director for store checkpoint files
    if os.path.exists(args.modeldir) is not True:
        os.mkdir(args.modeldir)
    stats_ = stats(args.modeldir, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.GSoP_mode)
        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)
        stats_.trainObj.append(trainObj)
        stats_.trainTop1.append(top1.cpu().numpy())
        stats_.trainTop5.append(top5.cpu().numpy())
        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion)
        stats_.valObj.append(valObj)
        stats_.valTop1.append(prec1.cpu().numpy())
        stats_.valTop5.append(prec5.cpu().numpy())
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)
        plot_cruve(stats_, args.modeldir, True)
        data = stats_
        sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
        

def plot_cruve(stats, path, iserr):
    trainObj = np.array(stats.trainObj)
    valObj = np.array(stats.valObj)
    if iserr:
        trainTop1 = 100 - np.array(stats.trainTop1)
        trainTop5 = 100 - np.array(stats.trainTop5)
        valTop1 = 100 - np.array(stats.valTop1)
        valTop5 = 100 - np.array(stats.valTop5)
        titleName = 'error'
    else:
        trainTop1 = np.array(stats.trainTop1)
        trainTop5 = np.array(stats.trainTop5)
        valTop1 = np.array(stats.valTop1)
        valTop5 = np.array(stats.valTop5)
        titleName = 'accuracy'
    epoch = len(trainObj)
    figure = plt.figure()
    obj = plt.subplot(1,3,1)
    obj.plot(range(1,epoch+1),trainObj,'o-',label = 'train')
    obj.plot(range(1,epoch+1),valObj,'o-',label = 'val')
    plt.xlabel('epoch')
    plt.title('objective')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(1,3,2)
    top1.plot(range(1,epoch+1),trainTop1,'o-',label = 'train')
    top1.plot(range(1,epoch+1),valTop1,'o-',label = 'val')
    plt.title('top1'+titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])
    top5 = plt.subplot(1,3,3)
    top5.plot(range(1,epoch+1),trainTop5,'o-',label = 'train')
    top5.plot(range(1,epoch+1),valTop5,'o-',label = 'val')
    plt.title('top5'+titleName)
    plt.xlabel('epoch')
    handles, labels = top5.get_legend_handles_labels()
    top5.legend(handles[::-1], labels[::-1])
    filename = os.path.join(path, 'net-train.pdf')
    figure.savefig(filename, bbox_inches='tight')
    
def vizNet(model, path):
    x = torch.randn(1,3,224,224)
    y = model(x)
    g = make_dot(y)
    g.render(os.path.join(path,'graph'), view=False)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, GSoP_mode):

    if GSoP_mode == 1:
        if epoch < 30:
            lr = args.lr
        elif epoch < 60:
            lr = args.lr*0.1
        elif epoch < 90:
            lr = args.lr*0.01
        else:
            lr = args.lr*0.001

    elif GSoP_mode == 2:
        if epoch < 30:
            lr = args.lr
        elif epoch < 45:
            lr = args.lr*0.1
        elif epoch < 60:
            lr = args.lr*0.01
        else:
            lr = args.lr*0.001

    print('the learning rate is set to {0:.5f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

