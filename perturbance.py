import torch
import pdb

def stochastic_label_perturbation(gt, alpha, beta, noise=False):
    new_label = torch.zeros(gt.shape).cuda()
    prob = torch.rand(gt.shape[:1]).cuda()
    new_label[prob >= alpha] = gt[prob >= alpha]
    new_label[prob < alpha] = (1 - 2 * beta) * gt[prob < alpha] + beta
    if noise == True:
        e = torch.empty_like(gt[prob < alpha])
        torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-0.25, b=0.25)
        new_label[prob < alpha] = new_label[prob < alpha] + e
    return new_label

def RS_ASLP(gt, alpha, beta, noise=False):
    new_label = torch.zeros(gt.shape).cuda()
    prob = torch.rand(gt.shape[:1]).cuda()
    new_label[prob >= alpha] = gt[prob >= alpha]
    new_label[prob < alpha] = (1 - 2 * beta) * gt[prob < alpha] + beta
    if noise == True:
        e = torch.empty_like(gt[prob < alpha])
        torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-0.25, b=0.25)
        new_label[prob < alpha] = new_label[prob < alpha] + e
    return new_label

def label_smoothing(gt, beta):
    n, c, h, w = gt.shape
    beta = beta.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
    b = torch.ones_like(gt) * beta
    new_label = (1 - 2 * b) * gt + b
    return new_label

def RS_label_smoothing(gt, scale):
    new_label = (1 - 2 * scale) * gt + scale
    return new_label

def ASLP_RS_label_smoothing(gt, scale, beta):
    _, _, h, w = gt.shape
    beta = beta.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    beta = beta.repeat(1, 1, h, w)
    scale = scale.cuda()
    new_label = (1 - 2 * beta) * gt + beta
    new_label = (1 - 2 * scale) * gt + scale
    return new_label
