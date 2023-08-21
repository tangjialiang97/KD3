from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import copy
from .util import AverageMeter, accuracy, reduce_tensor
from dataset.augmentation import MixDistribution
from distiller_zoo.relation import Contrastive_loss_bn

scale = nn.Sigmoid()
ce_loss = nn.CrossEntropyLoss(reduction='none')

def min_max(v):
    min_v = v.min()
    max_v = v.max()
    n1 = (v - min_v) / (max_v - min_v)
    return 1 - n1

def stand(v):
    mean_a = torch.mean(v)
    std_a = torch.std(v)
    n1 = (v - mean_a) / std_a
    print(mean_a)
    print(std_a)
    return 1.0 - n1

def weight_mse_loss(input, target, weights):
    weights = torch.tensor(weights)
    t_size = input.size(0)
    pct_var = (input - target) ** 2
    pct_var_temp = pct_var.clone()

    for i in range(t_size):
        pct_var_temp[i] = pct_var[i] * weights[i]

    loss = pct_var_temp.mean()
    return loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        if opt.dali is None:
            images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)



        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    embed_s = module_list[2]
    embed_t = module_list[3]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mix = MixDistribution(p=1.0, alpha=0.1, mix='random')

    pdist = nn.PairwiseDistance(p=2)

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    end = time.time()
    for idx, data in enumerate(train_loader):

        images, labels, _ = data
        images_aug = mix(images)

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            images_aug = images_aug.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        feat_s, _ = model_s(images, is_feat=True)
        feat_s_aug, _ = model_s(images_aug, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t_aug, logit_t_aug = model_t(images_aug, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            feat_t_aug = [f.detach() for f in feat_t_aug]

        cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
        model_t.get_feat_modules()[-1]


        trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
        logit_s = pred_feat_s

        dist = pdist(logit_s, logit_t)
        dist = torch.tensor(dist)
        dist_norm = scale(dist)
        loss_kd = weight_mse_loss(trans_feat_s, trans_feat_t, dist_norm.detach())


        xs = embed_s(feat_s[-1])
        xs_aug = embed_s(feat_s_aug[-1])
        xt = embed_t(feat_t[-1])
        xt_aug = embed_t(feat_t_aug[-1])
        loss_pro_s = Contrastive_loss_bn(xs, xs_aug, t=opt.temp)
        loss_pro_t = Contrastive_loss_bn(xt, xt_aug, t=opt.temp)

        loss_pro = loss_pro_s + loss_pro_t

        loss = loss_kd + loss_pro * opt.mix
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss_KD {loss_kd:.4f}\t'
                  'Loss_pro {loss_pro:.4f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss_kd=loss_kd, loss_pro=loss_pro, loss=losses, top1=top1, top5=top5,
                batch_time=batch_time))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg

def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            feat_s, _ = model_s(images, is_feat=True)
            feat_t, _ = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
            model_t.get_feat_modules()[-1]
            _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)

            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg




