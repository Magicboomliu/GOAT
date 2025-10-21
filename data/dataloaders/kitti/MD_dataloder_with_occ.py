from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os

from goat.utils.core.utils import read_text_lines
from goat.utils.core.file_io import read_disp, read_img
from skimage import io, transform
import numpy as np

class Middleburydataset(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='MID',
                 mode='train',
                 save_filename=False,
                 transform=None):
        super(Middleburydataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
   

        MID_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        dataset_name_dict = {
            'MID': MID_dict
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()
            # Left Image & Right Image
            left_img, right_img = splits[:2]
            right_imgE =None if len(splits)<3 else splits[2]
            right_imgL = None if len(splits)<4 else splits[3]
            left_disp = None if len(splits)<5 else splits[4]
            right_disp = None if len(splits)<6 else splits[5]
            occlusion_mask = None if len(splits)<7 else splits[6]
            
            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, left_disp) if left_disp is not None else None
            sample['occu_left'] = os.path.join(data_dir,occlusion_mask) if occlusion_mask is not None else None
            

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['img_right'] = read_img(sample_path['right'])
    
    
        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['gt_disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
        
        if sample_path['occu_left'] is not None:
            sample['occu_left'] = np.load(sample_path['occu_left']) #[H,W]
         
        if self.mode=='test' or self.mode=='val':
            pass

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
