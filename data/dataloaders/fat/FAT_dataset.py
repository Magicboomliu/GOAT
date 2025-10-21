from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os

from goat.utils.core.utils import read_text_lines
from goat.utils.core.file_io import read_disp, read_img, depth2disp
from skimage import io, transform
import numpy as np


class FATDataset(Dataset):
    def __init__(self,data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='FAT',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(FATDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(540, 960)
        self.scale_size =(576,960)
        
        
        FAT_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        dataset_name_dict = {
            'FAT': FAT_finalpass_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name
        self.samples = []
        data_filenames = dataset_name_dict[dataset_name][mode]
        lines = read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()
            left_img, right_img = splits[:2] # left and right images
            left_depth = splits[2]           # left depth
            right_depth = splits[3]          # right depth
            left_occ = splits[4]             # left occ
            right_occ = splits[5]            # right occ
            
            sample = dict()
            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]
            
            # laode the images
            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['left_disp'] = os.path.join(data_dir,left_depth)
            # sample['right_disp'] = os.path.join(data_dir,right_depth)
            sample['left_occ'] = os.path.join(data_dir,left_occ)
            # sample['right_occ'] = os.path.join(data_dir,right_occ)
            
            
            self.samples.append(sample)


    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['img_right'] = read_img(sample_path['right'])
        
        sample['left_disp'] = depth2disp(sample_path['left_disp']) #[H,W]
        # sample['right_disp'] = depth2disp(sample_path['right_disp']) #[H,W]
        
        sample['occu_left'] = np.load(sample_path['left_occ']) #[H,W]
        # sample['occ_right'] = np.load(sample_path['right_occ']) #[H,W]
        
        

        if self.mode=='test' or self.mode=='val':
            img_left = transform.resize(sample['img_left'], [576,960], preserve_range=True)
            img_right = transform.resize(sample['img_right'], [576,960], preserve_range=True)
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)
            sample['img_left'] = img_left
            sample['img_right'] = img_right

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    
    
    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size