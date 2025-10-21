from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from utils import utils
from goat.utils.core.kitti_io import read_img, read_disp, read_kitti_step1, read_kitti_step2, read_kitti_image_step1, read_kitti_image_step2, read_occ
from skimage import io, transform
import numpy as np
from PIL import Image


class StereoDataset_KITTI_OCC(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='KITTI',
                 mode='train',
                 save_filename=False,
                 transform=None):
        super(StereoDataset_KITTI_OCC, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(384, 1280)
        self.scale_size =(384,1280)
        self.original_size =(375,1280)
        

        sceneflow_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'val': 'filenames/KITTI_2012_val.txt',
        }

        kitti_2015_dict = {
            'train': self.train_datalist,
            'test': self.test_datalist
        }

        kitti_mix_dict = {
            'train': self.train_datalist,
            'test': self.test_datalist
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]
            gt_occ = None if len(splits)==3 else splits[3]
            sample = dict()
            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None
            sample['occ'] = os.path.join(data_dir,gt_occ) if gt_occ is not None else None

            self.samples.append(sample)



    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        if self.mode=='train':
            sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
            sample['img_right'] = read_img(sample_path['right'])
        
        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['gt_disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
            w = sample['gt_disp'].shape[-1]
            sample['occu_left'] = read_occ(sample_path['occ']) # [H,W]
            


        if self.mode=='test' or self.mode=='val':
            # Image Crop Operation
            left_im = read_kitti_image_step1(sample_path['left']) #[H,W,3]
            right_im = read_kitti_image_step1(sample_path['right'])        
            w, h = left_im.size
            left_image = left_im.crop((w-1280, h-384, w, h))
            right_image = right_im.crop((w-1280, h-384, w, h))
            sample['img_left'] = read_kitti_image_step2(left_image)
            sample['img_right'] = read_kitti_image_step2(right_image)
            w1,h1 = left_image.size
            
            # Disparity Crop Operation
            gt_disp = read_kitti_step1(sample_path['disp'])
            w, h = gt_disp.size
            dataL = gt_disp.crop((w-1280, h-384, w, h))
            dataL = read_kitti_step2(dataL)
            sample['gt_disp']= dataL
            
            gt_occ = read_occ(sample_path['occ'])
            gt_occ_pil = Image.fromarray(gt_occ)
            w,h = gt_occ_pil.size
            gt_occ_pil = gt_occ_pil.crop((w-1280, h-384, w, h))
            gt_occ = np.array(gt_occ_pil)
            sample['occu_left'] = gt_occ
            
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size



