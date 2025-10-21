from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import cv2

class DataAugmentation(object):
    def __init__(self):
        self.random_brightness = np.random.uniform(0.8,1.2)
        self.random_contrast = np.random.uniform(0.8,1.2)
        self.random_gamma = np.random.uniform(0.8,1.2)
        self.rng = np.random.RandomState(0)
        self.min_scale = 1.0
        self.max_scale = 1.5
    
    def chromatic_augmentation(self,img):
        
        #
        img = Image.fromarray(np.uint8(img))
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, self.random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img).astype(np.float32)
        
        return img_
        
    def __call__(self, sample):
        
        # (1)chromatic augmentation
        sample['img_left'] = self.chromatic_augmentation(sample['img_left'])
        sample['img_right'] = self.chromatic_augmentation(sample['img_right'])
        cv2.ocl.setUseOpenCL(False) 
        cv2.setNumThreads(0)
        
        # (2) Vertical Shifted
        if self.rng.binomial(1, 0.5):
            pixel = 2
            px = self.rng.uniform(-pixel, pixel)
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            sample['img_right'] = cv2.warpAffine(
                sample['img_right'] , trans_mat, sample['img_right'].shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # 2.2) random resize
        resize_scale = self.rng.uniform(self.min_scale, self.max_scale)
        
        sample['img_left'] = cv2.resize(
            sample['img_left'],
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        
        sample['img_right'] = cv2.resize(
            sample['img_right'],
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        sample['gt_disp'] = (
            cv2.resize(
                sample['gt_disp'],
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )
        
        sample['occu_left'] = sample['occu_left'].astype(np.float32)
        # print(sample['gt_disp'].dtype)
        sample['occu_left'] = cv2.resize(
                sample['occu_left'],
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        sample['occu_left'] = sample['occu_left'].astype(np.bool)
        
        return sample
      
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
        if 'left_disp' in sample.keys():
            disp = sample['left_disp']  # [H, W]
            sample['left_disp'] = torch.from_numpy(disp)
        
        if 'occu_left' in sample.keys():
            occu_left = sample['occu_left'] #[H,W]
            sample['occu_left'] = torch.from_numpy(occu_left)
    
        return sample

class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['img_left', 'img_right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample

class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        

        ori_height, ori_width = sample['img_left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['img_left'] = np.lib.pad(sample['img_left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['img_right'] = np.lib.pad(sample['img_right'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'left_disp' in sample.keys():
                sample['left_disp'] = np.lib.pad(sample['left_disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)
            if 'occu_left' in sample.keys():
                sample['occu_left'] = np.lib.pad(sample['occu_left'],
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

            sample['img_left'] = self.crop_img(sample['img_left'])
            sample['img_right'] = self.crop_img(sample['img_right'])
            
            if 'left_disp' in sample.keys():
                sample['left_disp'] = self.crop_img(sample['left_disp'])
            if 'occu_left' in sample.keys():
                sample['occu_left'] = self.crop_img(sample['occu_left'])
        
        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

class RandomVerticalFlip(object):
    """Randomly vertically filps"""
    def __call__(self, sample):
        if np.random.random() < 0.09:
            sample['img_left'] = np.copy(np.flipud(sample['img_left']))
            sample['img_right'] = np.copy(np.flipud(sample['img_right']))

            sample['left_disp'] = np.copy(np.flipud(sample['left_disp']))
            if 'occu_left' in sample.keys():
                sample['occu_left'] = np.copy(np.flipud(sample['occu_left']))                

        return sample

class ToPILImage(object):
    def __call__(self, sample):
        sample['img_left'] = Image.fromarray(sample['img_left'].astype('uint8'))
        sample['img_right'] = Image.fromarray(sample['img_right'].astype('uint8'))

        return sample

class ToNumpyArray(object):
    def __call__(self, sample):
        sample['img_left'] = np.array(sample['img_left']).astype(np.float32)
        sample['img_right'] = np.array(sample['img_right']).astype(np.float32)

        return sample



# Random coloring
class RandomContrast(object):
    """Random contrast"""
    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['img_left'] = F.adjust_contrast(sample['img_left'], contrast_factor)
            sample['img_right'] = F.adjust_contrast(sample['img_right'], contrast_factor)

        return sample

class RandomGamma(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet
            sample['img_left'] = F.adjust_gamma(sample['img_left'], gamma)
            sample['img_right'] = F.adjust_gamma(sample['img_right'], gamma)

        return sample

class RandomBrightness(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['img_left'] = F.adjust_brightness(sample['img_left'], brightness)
            sample['img_right'] = F.adjust_brightness(sample['img_right'], brightness)

        return sample

class RandomHue(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['img_left'] = F.adjust_hue(sample['img_left'], hue)
            sample['img_right'] = F.adjust_hue(sample['img_right'], hue)

        return sample

class RandomSaturation(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['img_left'] = F.adjust_saturation(sample['img_left'], saturation)
            sample['img_right'] = F.adjust_saturation(sample['img_right'], saturation)
        
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