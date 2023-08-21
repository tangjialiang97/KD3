import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd


class MB(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def mb(inputs, indexes, features, momentum=0.5):
    return MB.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_samples, args, temp=0.05, momentum=0.2):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.args = args

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # the source-like samples labels
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, inputs_s, indexes, k=10):
        # inputs: B*hidden_dim, features: L*hidden_dim
        inputs_out = mb(inputs, indexes, self.features, self.momentum)
        inputs_out /= self.temp  # B*L

        # generate local information
        B = inputs.size(0)
        local = inputs.mm(inputs_s.t())  # B*B
        _, neibor_idx = torch.topk(inputs_out, k)  # B*k
        neibor_ftr = self.features[neibor_idx].permute(0, 2, 1)  # B*2048*k

        _local = (torch.bmm(inputs.unsqueeze(1), neibor_ftr)).sum(-1)  # B * 1
        local = (local + _local.expand_as(local)) * (torch.eye(B).cuda())
        local /= self.temp

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps / masked_sums)

        # Achieve adaptive contrastive learning
        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max() + 1, B).float().cuda()  # L * B
        sim.index_add_(0, labels, inputs_out.t().contiguous())
        # add the local information
        sim.index_add_(0, targets, local.contiguous())

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())

        nums_help = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums_help.index_add_(0, targets, torch.ones(B, 1).float().cuda())
        nums += (nums_help > 0).float() * (k + 1)
        # avoid divide 0
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)

        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())

        loss1 = F.nll_loss(torch.log(masked_sim + 1e-6), targets)

        return loss1
