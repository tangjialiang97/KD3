import torch
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		if label == 1:
			img = tensor_rot_90(img)
		elif label == 2:
			img = tensor_rot_180(img)
		elif label == 3:
			img = tensor_rot_270(img)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	elif label == 'expand':
		labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3])
		batch = batch.repeat((4,1,1,1))
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels


class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))

def rotate_loss_1(images, model_s, rot_head):

	images_rot = torch.cat([torch.rot90(images, i, [2, 3]) for i in range(4)], dim=0).cuda()
	target_r = torch.cat([torch.empty(images.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()
	feat_s_rot, _ = model_s(images_rot, is_feat=True)
	loss_rot = F.cross_entropy(rot_head(feat_s_rot[-1]), target_r, reduction='mean')
	return loss_rot

def rotate_loss_2(images, model_s, rot_head):
	criterion = nn.CrossEntropyLoss().cuda()
	inputs_ssh, labels_ssh = rotate_batch(images, 'rand')
	inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
	feat_s_rot, _ = model_s(inputs_ssh, is_feat=True)

	outputs_ssh = rot_head(feat_s_rot[-1])
	loss_ssh = criterion(outputs_ssh, labels_ssh)
	return loss_ssh