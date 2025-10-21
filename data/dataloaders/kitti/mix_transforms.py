from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        left = np.transpose(sample['img_left'], (2, 0, 1))  # [3, H, W]
        sample['img_left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['img_right'], (2, 0, 1))
        sample['img_right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'gt_disp' in sample.keys():
            
            disp = sample['gt_disp']  # [H, W]
            sample['gt_disp'] = torch.from_numpy(disp)
            
        if 'occu_left' in sample.keys():
            occu_left = sample['occu_left'] #[H,W]
            sample['occu_left'] = torch.from_numpy(occu_left)
        
        if 'gt_normal' in sample.keys():
            
            normal = np.transpose(sample['gt_normal'],(2,0,1))
            
            sample['gt_normal'] = torch.from_numpy(normal)

        if 'pseudo_disp' in sample.keys():
            disp = sample['pseudo_disp']  # [H, W]
            sample['pseudo_disp'] = torch.from_numpy(disp)

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['img_left_sf', 'img_right_sf','img_left_rw', 'img_right_rw']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample



class ToTensorMix(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        
        if 'img_left_sf'  in sample.keys():
            left = np.transpose(sample['img_left_sf'], (2, 0, 1))  # [3, H, W]
            sample['img_left_sf'] = torch.from_numpy(left) / 255.
        
        if 'img_right_sf' in sample.keys():
            right = np.transpose(sample['img_right_sf'], (2, 0, 1))
            sample['img_right_sf'] = torch.from_numpy(right) / 255.
            
        if 'img_left_rw' in sample.keys():
            left = np.transpose(sample['img_left_rw'], (2, 0, 1))  # [3, H, W]
            sample['img_left_rw'] = torch.from_numpy(left) / 255.
        if 'img_right_rw' in sample.keys():
            right = np.transpose(sample['img_right_rw'], (2, 0, 1))
            sample['img_right_rw'] = torch.from_numpy(right) / 255.
            
        if 'occu_left_sf' in sample.keys():
            occu_left = sample['occu_left_sf'] #[H,W]
            sample['occu_left_sf'] = torch.from_numpy(occu_left)

        if 'occu_left_rw' in sample.keys():
            occu_left = sample['occu_left_sf'] #[H,W]
            sample['occu_left_sf'] = torch.from_numpy(occu_left)
            
        if 'gt_disp_rw' in sample.keys():
            disp = sample['gt_disp_rw']  # [H, W]
            sample['gt_disp_rw'] = torch.from_numpy(disp)
        
        if 'gt_disp_sf' in sample.keys():
            disp = sample['gt_disp_sf']  # [H, W]
            sample['gt_disp_sf'] = torch.from_numpy(disp)

        return sample





class RandomCrop_SF(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['img_left_sf'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['img_left_sf'] = np.lib.pad(sample['img_left_sf'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['img_right_sf'] = np.lib.pad(sample['img_right_sf'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'gt_disp_sf' in sample.keys():
                sample['gt_disp_sf'] = np.lib.pad(sample['gt_disp_sf'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)
            if 'occu_left_sf' in sample.keys():
                sample['occu_left_sf'] = np.lib.pad(sample['occu_left_sf'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['img_left_sf'] = self.crop_img(sample['img_left_sf'])
            sample['img_right_sf'] = self.crop_img(sample['img_right_sf'])
            if 'gt_disp_sf' in sample.keys():
                sample['gt_disp_sf'] = self.crop_img(sample['gt_disp_sf'])
            if 'occu_left_sf' in sample.keys():
                sample['occu_left_sf'] = self.crop_img(sample['occu_left_sf'])

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]




class RandomCrop_RW(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['img_left_rw'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['img_left_rw'] = np.lib.pad(sample['img_left_rw'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['img_right_rw'] = np.lib.pad(sample['img_right_rw'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'gt_disp_rw' in sample.keys():
                sample['gt_disp_rw'] = np.lib.pad(sample['gt_disp_rw'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)
            if 'occu_left_rw' in sample.keys():
                sample['occu_left_rw'] = np.lib.pad(sample['occu_left_rw'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['img_left_rw'] = self.crop_img(sample['img_left_rw'])
            sample['img_right_rw'] = self.crop_img(sample['img_right_rw'])
            if 'gt_disp_rw' in sample.keys():
                sample['gt_disp_rw'] = self.crop_img(sample['gt_disp_rw'])
            if 'occu_left_rw' in sample.keys():
                sample['occu_left_rw'] = self.crop_img(sample['occu_left_rw'])

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]




import matplotlib.pyplot as plt

class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.09:
            sample['img_left_sf'] = np.copy(np.flipud(sample['img_left_sf']))
            sample['img_right_sf'] = np.copy(np.flipud(sample['img_right_sf']))

            sample['gt_disp_sf'] = np.copy(np.flipud(sample['gt_disp_sf']))
            if 'occu_left_sf' in sample.keys():
                sample['occu_left_sf'] = np.copy(np.flipud(sample['occu_left_sf']))

            sample['img_left_rw'] = np.copy(np.flipud(sample['img_left_rw']))
            sample['img_right_rw'] = np.copy(np.flipud(sample['img_right_rw']))

            if 'gt_disp_rw' in sample.keys():
                sample['gt_disp_rw'] = np.copy(np.flipud(sample['gt_disp_rw']))
            if 'occu_left_rw' in sample.keys():
                sample['occu_left_rw'] = np.copy(np.flipud(sample['occu_left_rw']))             

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['img_left_sf'] = Image.fromarray(sample['img_left_sf'].astype('uint8'))
        sample['img_right_sf'] = Image.fromarray(sample['img_right_sf'].astype('uint8'))
        sample['img_left_rw'] = Image.fromarray(sample['img_left_rw'].astype('uint8'))
        sample['img_right_rw'] = Image.fromarray(sample['img_right_rw'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['img_left_sf'] = np.array(sample['img_left_sf']).astype(np.float32)
        sample['img_right_sf'] = np.array(sample['img_right_sf']).astype(np.float32)
        sample['img_left_rw'] = np.array(sample['img_left_rw']).astype(np.float32)
        sample['img_right_rw'] = np.array(sample['img_right_rw']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['img_left_sf'] = F.adjust_contrast(sample['img_left_sf'], contrast_factor)
            sample['img_right_sf'] = F.adjust_contrast(sample['img_right_sf'], contrast_factor)
            sample['img_left_rw'] = F.adjust_contrast(sample['img_left_rw'], contrast_factor)
            sample['img_right_rw'] = F.adjust_contrast(sample['img_right_rw'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)  # adopted from FlowNet

            sample['img_left_sf'] = F.adjust_gamma(sample['img_left_sf'], gamma)
            sample['img_right_sf'] = F.adjust_gamma(sample['img_right_sf'], gamma)
            sample['img_left_rw'] = F.adjust_gamma(sample['img_left_rw'], gamma)
            sample['img_right_rw'] = F.adjust_gamma(sample['img_right_rw'], gamma)

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)

            sample['img_left_sf'] = F.adjust_brightness(sample['img_left_sf'], brightness)
            sample['img_right_sf'] = F.adjust_brightness(sample['img_right_sf'], brightness)
            sample['img_left_rw'] = F.adjust_brightness(sample['img_left_rw'], brightness)
            sample['img_right_rw'] = F.adjust_brightness(sample['img_right_rw'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            sample['img_left_sf'] = F.adjust_hue(sample['img_left_sf'], hue)
            sample['img_right_sf'] = F.adjust_hue(sample['img_right_sf'], hue)
            sample['img_left_rw'] = F.adjust_hue(sample['img_left_rw'], hue)
            sample['img_right_rw'] = F.adjust_hue(sample['img_right_rw'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['img_left_sf'] = F.adjust_saturation(sample['img_left_sf'], saturation)
            sample['img_right_sf'] = F.adjust_saturation(sample['img_right_sf'], saturation)
            sample['img_left_rw'] = F.adjust_saturation(sample['img_left_rw'], saturation)
            sample['img_right_rw'] = F.adjust_saturation(sample['img_right_rw'], saturation)
        
        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample