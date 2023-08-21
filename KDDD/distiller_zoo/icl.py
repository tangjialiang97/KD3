from __future__ import print_function

import torch.nn as nn
import torch

class IclLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(IclLoss, self).__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, feat_s, feat_t):
        batch_size = feat_s.size(0)
        temperature = 0.3
        sim = torch.mm(feat_s, feat_t.t())
        mask = (torch.ones_like(sim) - torch.eye(batch_size, device=sim.device)).bool()
        sim = sim.masked_select(mask).view(batch_size, -1)

        logits_pos = torch.bmm(feat_s.view(batch_size, 1, -1), feat_t.view(batch_size, -1, 1)).squeeze(-1)
        logits_neg = torch.cat([sim], dim=1)

        logits = torch.cat([logits_pos, logits_neg], dim=1)

        instance_labels = torch.zeros(batch_size).long().cuda()

        loss_instance = self.crit(logits / temperature, instance_labels)
        return loss_instance

