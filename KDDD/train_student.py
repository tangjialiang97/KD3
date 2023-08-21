"""
the general training framework
"""
from __future__ import print_function

import os
import re
import argparse
import time

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import KDDD
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.I32 import get_i32_dataloader, train_loader_select

from helper.loops import train_distill as train, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate, TimeConverter
from distiller_zoo.pro import Embed
from distiller_zoo import IclLoss
from helper.util import time_d

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--hidden', type=int, default=512, help='hidden size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--p_cutoff', type=float, default=0.95, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--model_s', type=str, default='resnet18_cifar')
    parser.add_argument('--model_t', type=str, default='resnet34_cifar')
    parser.add_argument('--path_t', type=str, default='./save/teachers/models/***/***.pth', help='teacher model snapshot')
    parser.add_argument('--data_path', type=str, default='./data/ImageNet-32.py')
    parser.add_argument('--distill', type=str, default='kd')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('--temp', type=float, default=0.30)

    # distillation
    parser.add_argument('--trial', type=str, default='0', help='trial id')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='weight balance for other losses')
    parser.add_argument('-m', '--mix', type=float, default=0.001, help='weight balance for other losses')
    parser.add_argument('--in_dim', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=128)

    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    opt = parser.parse_args()

    # set the path of model and tensorboard
    opt.model_path = './save/students/models_final'
    opt.tb_path = './save/students/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_Mix', '_{}', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.mix, opt.beta, opt.trial)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = opt.model_t
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()


def main():
    opt = parse_option()

    # ASSIGN CUDA_ID
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = 0
    opt.gpu_id = 0

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    # model
    n_cls = {
        'cifar10': 10,
        'cinic': 10,
        'cifar100': 100,
        'tinyimagenet': 200,
    }.get(opt.dataset, None)
    opt.num_classes = n_cls
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    if opt.dataset == 'cifar10':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'cinic':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'tinyimagenet':
        data = torch.randn(2, 3, 64, 64)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = IclLoss()

    s_n = feat_s[-2].shape[1]
    t_n = feat_t[-2].shape[1]
    model_kddd = KDDD(s_n=s_n, t_n=t_n, factor=opt.factor)
    criterion_kd = nn.MSELoss()
    module_list.append(model_kddd)
    trainable_list.append(model_kddd)
    embed_s = Embed(feat_s[-1].shape[1], opt.out_dim)
    embed_t = Embed(feat_t[-1].shape[1], opt.out_dim)
    module_list.append(embed_s)
    module_list.append(embed_t)
    trainable_list.append(embed_s)
    trainable_list.append(embed_t)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    module_list.append(model_t)

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    # dataloader

    i32_train_loader = get_i32_dataloader(opt)
    if opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
    elif opt.dataset == 'cifar100':

        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)


    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)

        if opt.dali is not None:
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    for epoch in range(1, opt.epochs + 1):
        s_time = time.time()
        alpha = time_d(epoch, max_epoch=120)
        i32_train_loader_select = train_loader_select(module_list, i32_train_loader, alpha, opt)
        torch.cuda.empty_cache()


        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, i32_train_loader_select, module_list, criterion_list, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            text = ' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc,
                                                                                        train_acc_top5, time2 - time1)
            print(text)


        print('GPU %d validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            text = ' ** Acc@1 {:.3f}, Acc@5 {:.3f} , Best Acc@5 {:.3f}'.format(test_acc, test_acc_top5, best_acc)
            print(text)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                state_sim = {
                    'epoch': epoch,
                    'model': model_kddd.state_dict(),
                    'best_acc': best_acc,
                }

                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                save_file_sim = os.path.join(opt.save_folder, '{}_best_kddd.pth'.format(opt.model_s))

                test_merics = {'test_loss': test_loss,
                               'test_acc': test_acc,
                               'test_acc_top5': test_acc_top5,
                               'epoch': epoch}

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)
                torch.save(state_sim, save_file_sim)

        e_time = time.time()
        t_time = e_time - s_time
        text = TimeConverter(t_time, epoch=epoch, epochs=opt.epochs)
        print(text)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        print('best accuracy:', best_acc)
        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters()) / 1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] = (time.time() - total_time) / 3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(save_state, params_json_path)


if __name__ == '__main__':
    main()
