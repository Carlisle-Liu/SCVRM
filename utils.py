import torch
import numpy as np
import random


def regularisation_loss(pred, gt, constant):
    variable = (gt - 0.5) * pred
    loss = torch.mean(variable * constant)
    return loss

def compute_prediction_mean(pred):
    pred = pred.detach()
    pred = torch.sigmoid(pred)
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
    pred = torch.abs(pred - 0.5) + 0.5

    return torch.mean(pred)

def compute_accuracy(pred, gt):
    N, C, H, W = pred.shape
    pred = pred.detach()
    pred = torch.sigmoid(pred)
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
    gt = gt.detach()
    denominator = C * H * W
    diff = torch.abs(pred - gt)
    numerator = torch.tensor(torch.numel(diff[diff < 0.5]))
    return numerator / (denominator * N)

def compute_entropy(pred):
    a = torch.tensor(4)
    pred = torch.sigmoid(pred)
    ent = - torch.log(pred * (1 - pred))
    return torch.mean(ent) - torch.log(a)

def compute_entropy_individual(pred):
    a = torch.tensor(4)
    pred = torch.sigmoid(pred)
    pred = pred.clamp(1e-7, 1 - 1e-7)
    ent = - torch.log(pred * (1 - pred))
    return torch.mean(ent, (1, 2, 3))

def compute_bce_individual(pred, gt):
    pred = torch.sigmoid(pred)
    pred = pred.clamp(1e-7, 1 - 1e-7)
    bce = - gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)
    return torch.mean(bce, (1, 2, 3))

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        if len(self.losses) == 0:
            return torch.tensor(0.0)
        else:
            return torch.mean(torch.stack(self.losses))