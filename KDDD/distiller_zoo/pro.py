import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def eval_initial(memory, loader, model_t, opt):
    """Initialize the memory bank after one epoch warm up"""
    model_t.eval()


    features = torch.zeros(memory.num_samples, memory.num_features).cuda()
    labels = torch.zeros(memory.num_samples).long().cuda()
    outputs = torch.zeros(memory.num_samples, opt.num_class).cuda()
    with torch.no_grad():
        for i, (images, _, idxs) in enumerate(loader):
            images = images.cuda()
            feat_t, logit_t = model_t(images, is_feat=True)
            features[idxs] = feat_t[-1]
            labels[idxs] = (opt.num_class + idxs).long().cuda()
            outputs[idxs] = torch.softmax(logit_t, dim=-1)

        for i in range(opt.num_class):
            rank_out = outputs[:, i]
            _, r_idx = torch.topk(rank_out, 10)
            labels[r_idx] = i

        memory.features = F.normalize(features, dim=1)
        memory.labels = labels
    del features, labels, outputs

class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

def cal_prot(model, tranloader, opt):
    prototypes = []
    model.eval()
    for i, (images, _, _) in enumerate(tranloader):
        images = Variable(images).cuda()
        with torch.no_grad():
            feat_t, logit_t = model(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            pred = logit_t.data.max(1)[1]

            feature = feat_t[-1].data.cpu().numpy()
            labels = pred.data.cpu().numpy()
            if i == 0:
                features = np.zeros((len(tranloader.dataset), feature.shape[1]), dtype='float32')
                targets = np.zeros((len(tranloader.dataset)), dtype='int')
            if i < len(tranloader) - 1:
                features[i * opt.batch_size: (i + 1) * opt.batch_size] = feature
                targets[i * opt.batch_size: (i + 1) * opt.batch_size] = labels
            else:
                # special treatment for final batch
                features[i * opt.batch_size:] = feature
                targets[i * opt.batch_size:] = labels
    for c in range(opt.num_class):
        prototype = features[np.where(targets == c)].mean(0)  # compute prototypes with pseudo-label
        prototypes.append(torch.Tensor(prototype))
    prototypes = torch.stack(prototypes).cuda()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes

def mixup_constrastive_loss(input, input_aug, target, model_s, model_t, embed_s, embed_t, prototypes, opt):
    alpha = 8
    temperature = 0.3
    batch_size = input.size(0)
    num_classes = opt.num_class
    L = np.random.beta(alpha, alpha)
    labels = torch.zeros(batch_size, num_classes).cuda().scatter_(1, target.view(-1, 1), 1)

    inputs = torch.cat([input, input_aug], dim=0)
    idx = torch.randperm(batch_size * 2)
    labels = torch.cat([labels, labels], dim=0)

    input_mix = L * inputs + (1 - L) * inputs[idx]
    labels_mix = L * labels + (1 - L) * labels[idx]
    if opt.type == 1:
        feat_t, _ = model_t(input_mix, is_feat=True)
        feat_s, _ = model_s(input_mix, is_feat=True)

        feat_mix_t = embed_t(feat_t[-1])
        feat_mix_s = embed_s(feat_s[-1])
        # feat_mix_t = F.normalize(feat_mix_t, dim=1)
        # feat_mix_s = F.normalize(feat_mix_s, dim=1)

        logits_proto_t = torch.mm(feat_mix_t, prototypes.t()) / temperature
        logits_proto_s = torch.mm(feat_mix_s, prototypes.t()) / temperature
        loss_proto_t = -torch.mean(torch.sum(F.log_softmax(logits_proto_t, dim=1) * labels_mix, dim=1))
        loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels_mix, dim=1))

        loss = loss_proto_t + loss_proto_s

    elif opt.type == 2:
        feat_s, _ = model_s(input_mix, is_feat=True)

        feat_mix_s = embed_s(feat_s[-1])
        # feat_mix_s = F.normalize(feat_mix_s, dim=1)

        logits_proto_s = torch.mm(feat_mix_s, prototypes.t()) / temperature
        loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels_mix, dim=1))

        loss = loss_proto_s

    return loss

def mixup_constrastive_loss_noembed(input, input_aug, target, model_s, model_t, embed_s, embed_t, prototypes, opt):
    alpha = 8
    temperature = 0.3
    batch_size = input.size(0)
    num_classes = opt.num_class
    L = np.random.beta(alpha, alpha)
    labels = torch.zeros(batch_size, num_classes).cuda().scatter_(1, target.view(-1, 1), 1)

    inputs = torch.cat([input, input_aug], dim=0)
    idx = torch.randperm(batch_size * 2)
    labels = torch.cat([labels, labels], dim=0)

    input_mix = L * inputs + (1 - L) * inputs[idx]
    labels_mix = L * labels + (1 - L) * labels[idx]

    feat_t, _ = model_t(input_mix, is_feat=True)
    feat_s, _ = model_s(input_mix, is_feat=True)
    if opt.type == 1:
        feat_mix_t = embed_t(feat_t[-1])
        feat_mix_s = embed_s(feat_s[-1])
        feat_mix_t = F.normalize(feat_mix_t, dim=1)
        feat_mix_s = F.normalize(feat_mix_s, dim=1)
    elif opt.type == 2:
        feat_mix_t = F.normalize(feat_t[-1], dim=1)
        feat_mix_s = F.normalize(feat_s[-1], dim=1)

    logits_proto_t = torch.mm(feat_mix_t, prototypes.t()) / temperature
    logits_proto_s = torch.mm(feat_mix_s, prototypes.t()) / temperature
    loss_proto_t = -torch.mean(torch.sum(F.log_softmax(logits_proto_t, dim=1) * labels_mix, dim=1))
    loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels_mix, dim=1))

    return loss_proto_t + loss_proto_s

def constrastive_loss(input, input_aug, target, model_s, model_t, embed_s, embed_t, prototypes):
    temperature = 0.3
    inputs = torch.cat([input, input_aug], dim=0)
    labels = torch.zeros(inputs.size(0), 10).cuda().scatter_(1, target.view(-1, 1), 1)

    feat_t, _ = model_t(inputs, is_feat=True)
    feat_s, _ = model_s(inputs, is_feat=True)

    feat_t = embed_t(feat_t[-1])
    feat_s = embed_s(feat_s[-1])

    logits_proto_t = torch.mm(feat_t, prototypes.t()) / temperature
    logits_proto_s = torch.mm(feat_s, prototypes.t()) / temperature
    loss_proto_t = -torch.mean(torch.sum(F.log_softmax(logits_proto_t, dim=1) * labels, dim=1))
    loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels, dim=1))

    return loss_proto_t + loss_proto_s

# def cal_prot(model, embed_t, tranloader, num_class, opt):
#     prototypes = []
#     model.eval()
#     for i, (images, _, labels) in enumerate(tranloader):
#         images, labels = Variable(images).cuda(), Variable(labels).cuda()
#         with torch.no_grad():
#             feat, _ = model(images, is_feat=True)
#             feature = embed_t(feat[-1])
#             feature = feature.data.cpu().numpy()
#             labels = labels.data.cpu().numpy()
#             if i == 0:
#                 features = np.zeros((len(tranloader.dataset), feature.shape[1]), dtype='float32')
#                 targets = np.zeros((len(tranloader.dataset)), dtype='int')
#             if i < len(tranloader) - 1:
#                 features[i * opt.batch_size: (i + 1) * opt.batch_size] = feature
#                 targets[i * opt.batch_size: (i + 1) * opt.batch_size] = labels
#             else:
#                 # special treatment for final batch
#                 features[i * opt.batch_size:] = feature
#                 targets[i * opt.batch_size:] = labels
#     for c in range(num_class):
#         prototype = features[np.where(targets == c)].mean(0)  # compute prototypes with pseudo-label
#         prototypes.append(torch.Tensor(prototype))
#     prototypes = torch.stack(prototypes).cuda()
#     prototypes = F.normalize(prototypes, p=2, dim=1)
#     # print(prototypes)
#     return prototypes


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Rot(nn.Module):
    """Rot module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Rot, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        # x = self.l2norm(x)
        return x