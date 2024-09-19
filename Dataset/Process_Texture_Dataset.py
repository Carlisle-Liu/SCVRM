import os
import shutil
import cv2
import numpy as np
from PIL import Image


def process_texture_dataset(name_file_dir='./Texture_Image_Names.txt', src_data_dir='./dtd', tgt_data_dir='./Test/DTD_Texture_500'):
    file = open(name_file_dir, 'r')
    file_names = file.readlines()
    file.close()

    if not os.path.exists(tgt_data_dir):
        os.makedirs(tgt_data_dir)
        os.makedirs(os.path.join(tgt_data_dir, 'Image'))
        os.makedirs(os.path.join(tgt_data_dir, 'GT'))

    src_img_dir = os.path.join(src_data_dir, 'images')
    tgt_img_dir = os.path.join(tgt_data_dir, 'Image')
    tgt_gt_dir = os.path.join(tgt_data_dir, 'GT')

    for name in file_names:
        name = name.rstrip('\n')
        group = name.split('_')[0]

        src_path = os.path.join(src_img_dir, group, name)
        tgt_path = os.path.join(tgt_img_dir, name)

        try:
            shutil.copy(src_path, tgt_path)
        except:
            print("Error occurred while copying the file")
        
        img = np.asarray(Image.open(tgt_path))
        gt = np.zeros_like(img)
        gt_path = os.path.join(tgt_gt_dir, name.split('.')[0] + '.png')
        cv2.imwrite(gt_path, gt)

        

if __name__ == '__main__':
    process_texture_dataset()