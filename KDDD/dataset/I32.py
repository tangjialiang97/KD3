"""
get data loaders
"""
from __future__ import print_function

import _pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter
from PIL import Image


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class I32_dataset(Dataset):
    def __init__(self, opt, transform, model='normal', pred=[]):

        self.transform = transform
        self.model = model
        train_folder = opt.data_path
        with open(train_folder, 'rb') as fo:
            dict = _pickle.load(fo, encoding='latin1')
            self.train_data = dict['train']['data']
            self.train_target = dict['train']['target']

        data_num = len(self.train_data)
        self.train_data = self.train_data.reshape((data_num, -1, 32, 32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))

        if self.model == 'select':
            self.train_data = self.train_data[pred]
            self.train_target = self.train_target[pred]


    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_target[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.train_data)


def get_i32_dataloader(opt):
    normalize = transforms.Normalize(mean=[0.480, 0.457, 0.409],
                                     std=[0.275, 0.271, 0.281])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    npy_data = I32_dataset(opt, train_transform)
    trainloader = DataLoader(
        dataset=npy_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
    return trainloader

def train_loader_select(module_list, data_train_loader, alpha, opt):
    for module in module_list:
        module.eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
        model_t.get_feat_modules()[-1]
    problist = []
    predlist = []

    classwise_acc = torch.zeros((opt.num_classes),).cuda()
    p_cutoff_cls = torch.zeros((opt.num_classes),).cuda()


    for i, (images, _, _) in enumerate(data_train_loader):
        images = images.cuda()
        feat_s, _ = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, outputs_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
        _, _, outputs_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)

        outputs = outputs_t * alpha + outputs_s * (1 - alpha)
        pseudo_label = torch.softmax(outputs.detach(), dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        problist.append(max_probs)
        predlist.append(max_idx)
    problist = torch.cat(problist, dim=0)
    predlist = torch.cat(predlist, dim=0)

    pseudo_counter = Counter(predlist.tolist())
    for i in range(opt.num_classes):
        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
        p_cutoff_cls[i] = opt.p_cutoff * (classwise_acc[i] / (2. - classwise_acc[i]))
    pos_idx = []

    for i in range(len(problist)):
        p_cutoff = p_cutoff_cls[predlist[i]]
        if problist[i] > p_cutoff:
            pos_idx.append(i)

    normalize = transforms.Normalize(mean=[0.480, 0.457, 0.409],
                                     std=[0.275, 0.271, 0.281])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = I32_dataset(opt, train_transform, model='select', pred=pos_idx)

    trainloader_select = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                                                     num_workers=opt.num_workers)

    return trainloader_select