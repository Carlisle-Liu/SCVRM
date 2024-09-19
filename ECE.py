import os
import torch
import math
import pdb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from PIL import Image
from tqdm import tqdm



def ECE_EW(dir_pred, dir_gt, param, dataset='CoCA', method='Baseline', n_bins=10, norm=False):
    """
    :param dir_pred: the directory of predictions which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dir_gt: the directory of groundtruth which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dataset: dataset name;
    :param method: method name;
    :param n_bins: number of bins to be used in the histogram
    :return:
    """
    if not os.path.exists(dir_pred):
        print("The predictions of method: ({}) for dataset: ({}) is not available.".format(method, dataset))
        return
    print("ECE computation and plotting; Dataset: {}; Method: {}".format(dataset, method))

    sns.set(font_scale=1.5)

    bin_boundaries = torch.linspace(0, 1, steps=n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    oracle_bar = torch.linspace(0.05, 0.95, n_bins)
    oracle_bar[:n_bins//2] = 0.0

    oracle_line = torch.linspace(0.0, 1.0, n_bins)

    acc = torch.zeros(n_bins)
    conf = torch.zeros(n_bins)
    bins = torch.zeros(n_bins)

    ece = torch.tensor(0.0)
    oe = torch.tensor(0.0)

    df = pd.DataFrame(columns=['Confidence', 'Accuracy'])

    img_names = sorted(os.listdir(dir_pred))

    total_samples = len(img_names)

    x_coords = np.zeros((n_bins, len(img_names)))
    y_coords = np.zeros((n_bins, len(img_names)))

    for i in tqdm(range(total_samples), desc='{}'.format(dataset)):
        pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        pred = pred / 255.0
        pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        pred_bi[pred_bi < 128] = 0
        pred_bi[pred_bi > 127] = 255
        gt = np.asarray(Image.open(os.path.join(dir_gt, img_names[i])).convert('L')).flatten()
        gt = torch.tensor(gt)
        confidences = torch.maximum(pred, 1.0 - pred)

        if pred.shape != gt.shape:
            continue

        for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            if len(confidences[in_bin]) > 0:
                conf[j] = conf[j] + torch.sum(confidences[in_bin])
                x_coords[j][i] = torch.mean(confidences[in_bin]).item()
                correct = pred_bi[in_bin] == gt[in_bin]
                acc[j] = acc[j] + len(correct.masked_select(correct == True))
                bins[j] = bins[j] + len(confidences[in_bin])
                y_coords[j][i] = len(correct.masked_select(correct == True)) / len(confidences[in_bin])


    n_total = torch.sum(bins)
    acc_total = 0

    for k in range(len(acc)):
        acc_total += acc[k]

        acc[k] = acc[k] / (bins[k] + 1e-6)
        conf[k] = conf[k] / (bins[k] + 1e-6)
        ece += (torch.abs(conf[k] - acc[k])) * (bins[k] / n_total)
        if conf[k] > acc[k]:
            oe += (torch.abs(conf[k] - acc[k]) * conf[k]) * (bins[k] / n_total)


        if bins[k].item() > 0.0:
            cur_num = int(total_samples * bins[k].item() / n_total)
            mean_x_corrds = conf[k]
            mean_y_coords = acc[k]
            for q in range(cur_num):
                x_cor = mean_x_corrds
                y_cor = mean_y_coords
                new_row = pd.DataFrame({'Confidence': [x_cor.item()], 'Accuracy': [y_cor.item()]})
                df = df.append(new_row, ignore_index=True)


    if norm:
        print('Method: {}, Dataset: {}, ECE: {}, OE: {}, ACC: {}'.format(method + '_Min_Max_Norm', dataset, ece, oe, acc_total / n_total))
    else:
        print('Method: {}, Dataset: {}, ECE: {}, OE: {}, ACC: {}'.format(method, dataset, ece, oe, acc_total / n_total))
    graph = sns.jointplot(data=df, x='Confidence', y='Accuracy', xlim=[0.0, 1.0], ylim=[0.0, 1.0], kind='kde',
                          cmap='YlGnBu', fill=True, n_level=100)
    plt.plot(oracle_line, oracle_line, color='r', linestyle='dashed', linewidth=1)
    plt.grid()
    if not os.path.exists('./ECE/{}'.format(param.exp_group)):
        os.makedirs('./ECE/{}'.format(param.exp_group))
    if not os.path.exists('./ECE/{}/{}'.format(param.exp_group, param.exp_name)):
        os.makedirs('./ECE/{}/{}'.format(param.exp_group, method))
    if norm:
        plt.savefig('./ECE/{}/{}/Joint_{}_Min_Max_Norm.png'.format(param.exp_group, method, dataset), bbox_inches='tight', dpi=300)
    else:
        plt.savefig('./ECE/{}/{}/Joint_{}.png'.format(param.exp_group, method, dataset), bbox_inches='tight', dpi=300)
    plt.close()

    acc = acc_total / n_total

    return acc.item()*100, ece.item()*100, oe.item()*100


def Compute_ECE_Valid(dir_pred, dir_gt, param, n_bins=10, ACC=False):
    """
    :param dir_pred: the directory of predictions which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dir_gt: the directory of groundtruth which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param n_bins: number of bins to be used in the histogram
    :return:
    """
    bin_boundaries = torch.linspace(0, 1, steps=n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    oracle_bar = torch.linspace(0.05, 0.95, 10)
    oracle_bar[:5] = 0.0

    acc = torch.zeros(n_bins)
    conf = torch.zeros(n_bins)
    bins = torch.zeros(n_bins)

    ece = 0
    oe = 0

    img_names = sorted(os.listdir(dir_pred))

    for i in range(len(img_names)):
        pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i]))).flatten())
        pred = pred / 255.0
        pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i]))).flatten())
        pred_bi[pred_bi < 128] = 0
        pred_bi[pred_bi > 127] = 255
        gt = np.asarray(Image.open(os.path.join(dir_gt, img_names[i])).convert('L')).flatten()
        gt = torch.tensor(gt)

        confidences = torch.maximum(pred, 1.0 - pred)

        for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            if len(confidences[in_bin]) > 0:
                conf[j] = conf[j] + torch.sum(confidences[in_bin])
            bins[j] = bins[j] + len(confidences[in_bin])
            correct = pred_bi[in_bin] == gt[in_bin]
            acc[j] = acc[j] + len(correct.masked_select(correct == True))

    n_total = torch.sum(bins)

    acc_total = 0

    for k in range(len(acc)):
        acc_total += acc[k]
        acc[k] = acc[k] / (bins[k] + 1e-6)
        conf[k] = conf[k] / (bins[k] + 1e-6)
        ece += (conf[k] - acc[k]) * (bins[k] / n_total)

    if ACC:
        return acc_total / n_total

    return ece




def convert_predictions_to_1d_array(dir_pred, param, dataset, method, save=False, proportion=0.1):
    if not os.path.exists(dir_pred):
        print("The predictions of method: ({}) for dataset: ({}) is not available.".format(method, dataset))
        return

    img_names = sorted(os.listdir(dir_pred))
    total_samples = len(img_names)

    for i in range(total_samples):
        pred = np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten()
        pred = pred
        pred_inv = 255 - pred
        pred = np.maximum(pred, pred_inv)

        selected = np.random.choice(pred, int(pred.shape[0] * proportion))

        if i == 0:
            results = pred
        else:
            results = np.concatenate((results, selected), axis=None)

    if save == True:
        if not os.path.exists('./ECE/{}'.format(param.exp_group)):
            os.makedirs('./ECE/{}'.format(param.exp_group))
        if not os.path.exists('./ECE/{}/{}'.format(param.exp_group, param.exp_name)):
            os.makedirs('./ECE/{}/{}'.format(param.exp_group, method))
        np.save('./ECE/{}/{}/Numpy_Array_{}.npy'.format(param.exp_group, method, dataset), results)

    return results




def ECE_EM(dir_pred, dir_gt, param, dataset, method, n_bins):
    if not os.path.exists(dir_pred):
        print("The predictions of method: ({}) for dataset: ({}) is not available.".format(method, dataset))
        return
    pred = convert_predictions_to_1d_array(dir_pred=dir_pred, param=param, dataset=dataset, method=method)

    if np.size(pred) == 0:
        return np.linspace(0, 1, n_bins+1)[:-1]

    edge_indices = np.linspace(0, len(pred), n_bins, endpoint=False)
    edge_indices = np.round(edge_indices).astype(int)

    edges = np.sort(pred)[edge_indices]

    dict_edge = {}
    for e in edges:
        if e not in dict_edge:
            dict_edge[e] = 1
        else:
            dict_edge[e] += 1

    dict_edge[0] = 1

    bin_lowers = np.concatenate((np.array([0]), edges[:-1]))
    bin_uppers = np.concatenate((edges[1:], np.array([255])))

    acc = torch.zeros(n_bins)
    conf = torch.zeros(n_bins)
    bins = torch.zeros(n_bins)

    ece = 0
    oe = 0

    img_names = sorted(os.listdir(dir_pred))

    total_samples = len(img_names)

    for i in range(total_samples):
        pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        pred = pred
        pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        pred_bi[pred_bi < 128] = 0
        pred_bi[pred_bi > 127] = 255
        gt = np.asarray(Image.open(os.path.join(dir_gt, img_names[i])).convert('L')).flatten()
        gt = torch.tensor(gt)

        confidences = torch.maximum(pred, 255 - pred)

        if pred.shape != gt.shape:
            continue

        last_bin_lower = -1
        count = 0

        for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = confidences.ge(bin_lower) * confidences.le(bin_upper)
            if len(confidences[in_bin]) > 0:
                if bin_lower == last_bin_lower:
                    count += 1
                rep = dict_edge[bin_lower]
                numbers = int(math.floor(len(confidences[in_bin]) / rep))
                conf[j] = conf[j] + torch.sum(confidences[in_bin][count*numbers:(count+1)*numbers] / 255.0)
                correct = pred_bi[in_bin][count*numbers:(count+1)*numbers] == gt[in_bin][count*numbers:(count+1)*numbers]
                acc[j] = acc[j] + len(correct.masked_select(correct == True))
                bins[j] = bins[j] + len(confidences[in_bin][count*numbers:(count+1)*numbers])

    n_total = torch.sum(bins)
    acc_total = 0

    for k in range(len(acc)):
        acc_total += acc[k]
        acc[k] = acc[k] / (bins[k] + 1e-6)
        conf[k] = conf[k] / (bins[k] + 1e-6)
        ece += (torch.abs(conf[k] - acc[k])) * (bins[k] / n_total)
        if conf[k] > acc[k]:
            oe += (torch.abs(conf[k] - acc[k]) * conf[k]) * (bins[k] / n_total)

    print('Method: {}, Dataset: {}, ECE_EM: {}, OE_EM: {}, ACC: {}'.format(method, dataset, ece, oe, acc_total / n_total))





def ECE_SWEEP(dir_pred, dir_gt, param, dataset, method, n_bins):
    if not os.path.exists(dir_pred):
        print("The predictions of method: ({}) for dataset: ({}) is not available.".format(method, dataset))
        return
    pred_array = convert_predictions_to_1d_array(dir_pred=dir_pred, param=param, dataset=dataset, method=method)

    cur_oe = 0
    cur_ece = 0
    cur_acc_total = 0

    img_names = sorted(os.listdir(dir_pred))
    total_samples = len(img_names)

    pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[0])).convert('L')).flatten())
    pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[0])).convert('L')).flatten())
    gt = torch.tensor(np.asarray(Image.open(os.path.join(dir_gt, img_names[0])).convert('L')).flatten())

    for i in range(1, total_samples):
        cur_pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        cur_pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        cur_gt = torch.tensor(np.asarray(Image.open(os.path.join(dir_gt, img_names[i])).convert('L')).flatten())

        if cur_pred.shape != cur_gt.shape:
            continue
        else:
            pred = torch.concatenate((pred, cur_pred))
            pred_bi = torch.concatenate((pred_bi, cur_pred_bi))
            gt = torch.concatenate((gt, cur_gt))

    pred_bi[pred_bi < 128] = 0
    pred_bi[pred_bi > 127] = 255

    confidences = torch.maximum(pred, 255 - pred)

    for b in range(1, n_bins, 1):
        if np.size(pred_array) == 0:
            return np.linspace(0, 1, b+1)[:-1]

        edge_indices = np.linspace(0, len(pred_array), b, endpoint=False)
        edge_indices = np.round(edge_indices).astype(int)

        edges = np.sort(pred_array)[edge_indices]

        dict_edge = {}
        for e in edges:
            if e not in dict_edge:
                dict_edge[e] = 1
            else:
                dict_edge[e] += 1

        dict_edge[0] = 1
        bin_lowers = np.array([0])
        bin_uppers = np.array([255])
        bin_lowers = np.concatenate((bin_lowers, edges[:-1]))
        bin_uppers = np.concatenate((edges[1:], bin_uppers))

        acc = torch.zeros(b)
        conf = torch.zeros(b)
        bins = torch.zeros(b)

        ece = 0
        oe = 0

        last_bin_lower = -1
        count = 0

        for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = confidences.ge(bin_lower) * confidences.le(bin_upper)
            if len(confidences[in_bin]) > 0:
                if bin_lower == last_bin_lower:
                    count += 1
                rep = dict_edge[bin_lower]
                numbers = int(math.floor(len(confidences[in_bin]) / rep))
                conf[j] = conf[j] + torch.sum(confidences[in_bin][count * numbers:(count + 1) * numbers] / 255.0)
                correct = pred_bi[in_bin][count * numbers:(count + 1) * numbers] == gt[in_bin][count * numbers:(count + 1) * numbers]
                acc[j] = acc[j] + len(correct.masked_select(correct == True))
                bins[j] = bins[j] + len(confidences[in_bin][count * numbers:(count + 1) * numbers])

        n_total = torch.sum(bins)
        acc_total = 0

        for k in range(len(acc)):
            acc_total += acc[k]
            acc[k] = acc[k] / (bins[k] + 1e-6)
            conf[k] = conf[k] / (bins[k] + 1e-6)
            ece += (torch.abs(conf[k] - acc[k])) * (bins[k] / n_total)
            if conf[k] > acc[k]:
                oe += (torch.abs(conf[k] - acc[k]) * conf[k]) * (bins[k] / n_total)

        mono = acc[1:] - acc[:-1]
        if len(mono[mono < 0]) > 0:
            break

        cur_ece = ece
        cur_oe = oe
        cur_acc_total = acc_total

    print('Method: {}, Dataset: {}, ECE_SWEEP: {}, OE_SWEEP: {}, ACC: {}'.format(method, dataset, cur_ece, cur_oe, cur_acc_total / n_total))



def ECE_DEBIAS(dir_pred, dir_gt, param, dataset='CoCA', method='Baseline', n_bins=10):
    """
    :param dir_pred: the directory of predictions which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dir_gt: the directory of groundtruth which should be PNG images of uint8 data type. They should be loaded
    as image with H * W shape;
    :param dataset: dataset name;
    :param method: method name;
    :param n_bins: number of bins to be used in the histogram
    :return:
    """
    if not os.path.exists(dir_pred):
        print("The predictions of method: ({}) for dataset: ({}) is not available.".format(method, dataset))
        return

    bin_boundaries = torch.linspace(0, 1, steps=n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    acc = torch.zeros(n_bins)
    conf = torch.zeros(n_bins)
    bins = torch.zeros(n_bins)

    ece = 0
    oe = 0

    img_names = sorted(os.listdir(dir_pred))
    total_samples = len(img_names)

    for i in range(total_samples):
        pred = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        pred = pred / 255.0
        pred_bi = torch.tensor(np.asarray(Image.open(os.path.join(dir_pred, img_names[i])).convert('L')).flatten())
        pred_bi[pred_bi < 128] = 0
        pred_bi[pred_bi > 127] = 255
        gt = np.asarray(Image.open(os.path.join(dir_gt, img_names[i])).convert('L')).flatten()
        gt = torch.tensor(gt)

        confidences = torch.maximum(pred, 1.0 - pred)

        if pred.shape != gt.shape:
            continue

        for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            if len(confidences[in_bin]) > 0:
                conf[j] = conf[j] + torch.sum(confidences[in_bin])
                correct = pred_bi[in_bin] == gt[in_bin]
                acc[j] = acc[j] + len(correct.masked_select(correct == True))
                bins[j] = bins[j] + len(confidences[in_bin])


    n_total = torch.sum(bins)

    acc_total = 0

    for k in range(len(acc)):
        acc_total += acc[k]
        acc[k] = acc[k] / (bins[k] + 1e-6)
        conf[k] = conf[k] / (bins[k] + 1e-6)
        ece += (bins[k] / n_total) * ((conf[k] - acc[k])**2 - (conf[k] * (1 - conf[k])) / (bins[k] - 1))
        if conf[k] > acc[k]:
            oe += (bins[k] / n_total) * ((conf[k] - acc[k])**2 - (conf[k] * (1 - conf[k])) / (bins[k] - 1))

    print('Method: {}, Dataset: {}, ECE_DEBIAS: {}, OE_DEBIAS: {}, ACC: {}'.format(method, dataset, ece, oe, acc_total / n_total))

