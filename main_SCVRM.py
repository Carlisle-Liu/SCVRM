import argparse
import os
import cv2
import time
import torch
import torch.nn.functional as F
import pickle
import shutil
from tqdm import tqdm
import pandas as pd

from model.ResNet_models import Model
from data import Train_Loader, Test_Loader
from utils import adjust_lr, AvgMeter
from perturbance import *
from ECE import *
from SOD_Evaluation_Tool.evaluator import Eval_thread
from SOD_Evaluation_Tool.dataloader import EvalDataset

from radiation_function import Radiation_Function

import pdb


def train_GALAXY(param):
    generator = Model(channel=param.feat_channel)
    generator.cuda()
    generator_params = generator.parameters()
    generator_optimiser = torch.optim.Adam(generator_params, param.lr_gen)

    train_loader = Train_Loader(param.train_image_root,
                                param.train_gt_root,
                                batchsize = param.batchsize,
                                trainsize = param.trainsize,
                                name_file = param.train_name_file)
    
    RF = Radiation_Function(param.radiation)

    for epoch in range(1, (param.epoch + 1)):
        generator.train()
        loss_record = AvgMeter()

        with tqdm(train_loader, unit='batch') as pbar:
            for pack in pbar:
                pbar.set_description('Epoch {}/{}'.format(epoch, param.epoch))
                generator_optimiser.zero_grad()
                images, gts, names = pack

                gts = RS_label_smoothing(gts, param.smooth_base)

                n, c, h, w = images.shape
            
                imgs_copy = images.detach().clone()
                gts_copy = gts.detach().clone()

                images_2 = images.detach().clone()
                gts_2 = gts.detach().clone()

                for k in range(3):
                    images_k = imgs_copy.clone().detach()
                    noise = torch.rand_like(images_k)
                    scale = torch.rand(noise.shape[0])
                    sigma = scale * param.noise_sd
                    sigma = sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma = sigma.repeat(1, c, h, w)
                    images_k = images_k + noise * sigma
                    images = torch.cat((images, images_k), dim = 0)

                    gts_k = gts_copy.clone().detach()
                    smoothing_factor = RF.radiate(scale / param.eta)
                    inversed_smoothing_scale = param.smooth_radiation * (1 - scale)**2
                    smoothing_factor += inversed_smoothing_scale 
                    gts_k = label_smoothing(gts_k, smoothing_factor)
                    gts = torch.cat((gts, gts_k), dim = 0)

                images = images.cuda()
                gts = gts.cuda()

                pred = generator(images)

                loss_all = F.binary_cross_entropy_with_logits(pred, gts, reduce='none')

                loss_all.backward()
                generator_optimiser.step()

                # """SCVRM Code Here."""
                # for k in range(param.M % 3):
                #     images_k = imgs_copy.clone().detach()
                #     noise = torch.rand_like(images_k)
                #     scale = torch.rand(noise.shape[0])
                #     sigma = scale * param.noise_sd
                #     sigma = sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                #     sigma = sigma.repeat(1, c, h, w)
                #     images_k = images_k + noise * sigma
                #     images_2 = torch.cat((images_2, images_k), dim = 0)

                #     gts_k = gts_copy.clone().detach()
                #     smoothing_factor = RF.radiate(scale / param.eta)
                #     inversed_smoothing_scale = param.smooth_radiation * (1 - scale)**2
                #     smoothing_factor += inversed_smoothing_scale 
                #     gts_k = label_smoothing(gts_k, smoothing_factor)
                #     gts_2 = torch.cat((gts_2, gts_k), dim = 0)
            
                # images_2 = images_2.cuda()
                # gts_2 = gts_2.cuda()

                # pred_2 = generator(images_2)

                # loss_all = F.binary_cross_entropy_with_logits(pred_2, gts_2, reduce='none')

                # loss_all.backward()
                # generator_optimiser.step()

                loss_record.update(loss_all.data, n * (param.M % 3))
                pbar.set_postfix(Loss='{:.4f}'.format(loss_record.show().item()),
                                 lr=generator_optimiser.param_groups[0]['lr'])

        adjust_lr(generator_optimiser, epoch, param.decay_rate, param.decay_epoch)

        save_path = './experiments'
        exp_group_path = os.path.join(save_path, param.exp_group)
        exp_path = os.path.join(exp_group_path, param.exp_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(exp_group_path):
            os.makedirs(exp_group_path)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        if epoch % param.epoch == 0:
            torch.save(generator.state_dict(), exp_path + '/' + 'Model' + '_%d' % epoch + '_Gen.pth')



def acc_val(model, param):
    model = model.eval()
    save_path = './experiments/{}/{}/Validation/'.format(param.exp_group, param.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    valid_loader = Test_Loader(image_root = param.train_image_root, 
                               testsize = param.testsize, 
                               name_file = param.valid_name_file)

    for i in tqdm(range(valid_loader.size), desc='Validation'):
        image, HH, WW, name = valid_loader.load_data()
        image = image.cuda()
        generator_pred = model.forward(image)
        res = generator_pred
        res = F.interpolate(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * res
        cv2.imwrite(save_path + name, res)

    dir_pred = './experiments/{}/{}/{}'.format(param.exp_group, param.exp_name, 'Validation')
    acc = Compute_ECE_Valid(dir_pred = dir_pred, 
                            dir_gt = param.train_gt_root, 
                            param = pm, 
                            n_bins = 10, 
                            ACC = True)

    return acc



def eval(param):
    output_dir = './SOD_Evaluation_Tool/Result/Detail'
    pred_dir = './experiments/{}/{}/{}/'.format(param.exp_group, param.exp_name, param.exp_name)
    threads = []

    if param.test_dataset == 'All':
        test_datasets = ['DUTS-TE', 'DUT-OMRON', 'PASCAL-S', 'SOD', 'ECSSD', 'HKU-IS', 'DTD_Texture_500']
    else:
        test_datasets = [param.test_dataset]

    for dataset in test_datasets:
        loader = EvalDataset(os.path.join(pred_dir, dataset), os.path.join(pm.test_dataset_root, dataset, 'GT'))
        thread = Eval_thread(loader, param.exp_name, dataset, output_dir, True)
        threads.append(thread)
    for thread in threads:
        print(thread.run())



def test(param):
    generator = Model(channel=param.feat_channel)
    generator.load_state_dict(torch.load('./experiments/{}/{}/Model_{}_Gen.pth'.format(param.exp_group, param.exp_name, param.epoch)))
    generator.cuda()
    generator.eval()

    if param.test_dataset == 'All':
        test_datasets = ['DUTS-TE', 'DUT-OMRON', 'PASCAL-S', 'SOD', 'ECSSD', 'HKU-IS', 'DTD_Texture_500']
    else:
        test_datasets = [param.test_dataset]

    for dataset in test_datasets:
        print('Currently processing: {} dataset:'.format(dataset))
        save_path = './experiments/{}/{}/{}/'.format(param.exp_group, param.exp_name, param.exp_name) + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = param.test_dataset_root + dataset + '/Image/'
        test_loader = Test_Loader(image_root, param.testsize)
        for i in tqdm(range(test_loader.size), desc='{}'.format(dataset)):
            image, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            generator_pred = generator.forward(image)
            res = generator_pred
            res = F.interpolate(res, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            if dataset in ['PASCAL-S', 'ECSSD', 'HKU-IS']:
                res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            else:
                res = 255 * res
            cv2.imwrite(save_path + name, res)
        print('Testing in {} dataset has been completely!'.format(dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--testsize', type=int, default=384, help='testing dataset size')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='every n epochs decay learning rate')
    parser.add_argument('--feat_channel', type=int, default=256, help='reduced channel of saliency feat')
    parser.add_argument('--exp_group', type=str, default='Best_Model_for_Publication', help='experiment group name')
    parser.add_argument('--exp_name', type=str, default='LSR_GR_2.0_S_0.02_RS_0.05', help='experiment name')
    parser.add_argument('--bound', type=float, default=100.0, help='The clipping bound of data drawn from a Gaussian distribution')
    parser.add_argument('--noise_sd', type = float, default = 2.0, help = "standard deviation of Gaussian noise for data augmentation")
    parser.add_argument('--radiation', type = str, default = 'Exponential', help = 'Radiation Function of the Ball')
    parser.add_argument('--train_image_root', default='./Dataset/Train/DUTS-TR/Image/')
    parser.add_argument('--train_gt_root', default='./Dataset/Train/DUTS-TR/GT/')
    parser.add_argument('--train_name_file', default='./Dataset/DUTS-TR-Train.txt')
    parser.add_argument('--valid_name_file', default='./Dataset//DUTS-TR-Validation.txt')
    parser.add_argument('--test_dataset_root', default='./Dataset/Test/')
    parser.add_argument('--test_dataset', type=str, default = 'All')
    parser.add_argument('--smooth_base', type=float, default=0.02, help='Notation S')
    parser.add_argument('--smooth_radiation', type=float, default=0.05, help='Notation RS')
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--M', type=int, default=3)
    pm = parser.parse_args()
    print(pm)

    start = time.time()
    train_GALAXY(param=pm)
    end = time.time()
    duration = (start - end) / 3600
    print("Training time: {} hours".format(duration))

    test(pm)

    if pm.test_dataset == 'All':
        test_datasets = ['DUTS-TE', 'DUT-OMRON', 'PASCAL-S', 'SOD', 'ECSSD', 'HKU-IS', 'DTD_Texture_500']
    else:
        test_datasets = [pm.test_dataset]

    for dataset in test_datasets:
        dir_pred = './experiments/{}/{}/{}/{}'.format(pm.exp_group, pm.exp_name, pm.exp_name, dataset)
        dir_gt = './Dataset/Test/{}/GT'.format(dataset)
        acc, ece, oe = ECE_EW(dir_pred=dir_pred, 
                              dir_gt=dir_gt, 
                              param=pm, 
                              dataset=dataset, 
                              method=pm.exp_name, 
                              n_bins=10)
    
    eval(pm)

    dir_pred = './experiments/{}/{}/{}'.format(pm.exp_group, pm.exp_name, pm.exp_name)
    shutil.rmtree(dir_pred)

