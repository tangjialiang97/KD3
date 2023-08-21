"""
Functions to compute loss objectives of FedX.

"""

import torch

import torch.nn.functional as F

def Contrastive_loss(x1, x2, t=0.1):
    """Contrastive loss objective function"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    # print(sim_matrix)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # print(mask)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # print(sim_matrix)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def Contrastive_loss_bn(x1, x2, t=0.1):
    """Contrastive loss objective function"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    # print(sim_matrix)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # print(mask)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # print(sim_matrix)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def relation_loss(xs, xt, xs_random, xt_random, t=0.1, t2=0.1):
    """Relational loss objective function"""
    pred_sim1 = torch.mm(F.normalize(xs, dim=1), F.normalize(xs_random, dim=1).t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(F.normalize(xt, dim=1), F.normalize(xt_random, dim=1).t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0


# x1 = torch.randn(4, 4)
# x2 = torch.randn(4, 4)
# loss = Contrastive_loss(x1, x2)