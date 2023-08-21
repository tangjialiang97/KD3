import torch
import torch.nn.functional as F

def pcl(feats_s, feats_t, labels, prototypes, num_classes):

    temperature = 0.3
    batch_size = feats_s.size(0)

    labels = torch.zeros(batch_size, num_classes).cuda().scatter_(1, labels.view(-1, 1), 1)
    logits_proto_t = torch.mm(feats_t, prototypes.t()) / temperature
    logits_proto_s = torch.mm(feats_s, prototypes.t()) / temperature

    loss_proto_t = -torch.mean(torch.sum(F.log_softmax(logits_proto_t, dim=1) * labels, dim=1))
    loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels, dim=1))
    return loss_proto_t + loss_proto_s