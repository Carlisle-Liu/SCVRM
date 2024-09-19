import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np

def random_flip(img, label):
    flip = random.randint(0, 1)
    if flip == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def random_crop(img, label):
    border = 30
    image_width = img.size[0]
    image_height = img.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, 
        (image_height - crop_win_height) >> 1, 
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1
        )
    return img.crop(random_region), label.crop(random_region)

def random_rotation(img, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return img, label

class SOD_Dataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, name_file):
        self.trainsize = trainsize
        with open(name_file, 'r') as f:
            self.names = f.readlines()
        self.images = [image_root + f.rstrip('\n') + '.jpg' for f in self.names]
        self.gts = [gt_root + f.rstrip('\n') + '.png' for f in self.names]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        name = self.names[index]
        image, gt = random_flip(image, gt)
        image, gt = random_crop(image, gt)
        image, gt = random_rotation(image, gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt, name

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def Train_Loader(image_root, gt_root, name_file, batchsize, trainsize, shuffle = True, num_workers=12, pin_memory = True):
    dataset = SOD_Dataset(image_root, gt_root, trainsize, name_file)
    data_loader = data.DataLoader(dataset = dataset,
                                  batch_size = batchsize,
                                  shuffle = shuffle,
                                  num_workers = num_workers,
                                  pin_memory = pin_memory)
    return data_loader


class Test_Loader:
    def __init__(self, image_root, testsize, name_file=None):
        self.testsize = testsize
        if name_file:
            with open(name_file, 'r') as f:
                self.names = f.readlines()
            self.images = [image_root + f.rstrip('\n') + '.jpg' for f in self.names]
        else:
            self.images = [image_root + f for f in os.listdir(image_root)  if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')




def post_RS_normalisation(input):
    transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    input = transform(input)
    return input