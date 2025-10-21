import torch
import torch.nn as nn
import torch.nn.functional as F
from goat.utils.core.Color_Conversion.lab import lab_to_rgb, rgb_to_lab


# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Compute the Mean and Variance
def compute_mean_and_variance(tensor):
    channel_L = tensor[:,:1,:,:]
    channel_A = tensor[:,1:2,:,:]
    channel_B = tensor[:,2:3,:,:]
    
    LAB_mean = [channel_L.mean(),channel_A.mean(),channel_B.mean()]
    LAB_std = [channel_L.std(),channel_A.std(),channel_B.std()]
    LAB_mean = torch.Tensor(LAB_mean).type_as(tensor).view(1,3,1,1)
    LAB_mean.requires_grad = False
    LAB_std = torch.Tensor(LAB_std).type_as(tensor).view(1,3,1,1)
    LAB_std.requires_grad = False
    
    return LAB_mean,LAB_std


# dataset domain adaption
def dataset_domain_adaption(sytheic_dataset,real_world_dataset,cur_mean,cur_std,gamma=0.95):
    # Convert to a sythetic and real world dataset from RGB to LAB
    sytheic_data_lab = rgb_to_lab(sytheic_dataset)
    real_world_data_lab = rgb_to_lab(real_world_dataset)
    sytheic_data_lab_mean, sytheic_data_lab_std = compute_mean_and_variance(sytheic_data_lab)
    real_world_data_lab_mean, real_world_data_lab_std = compute_mean_and_variance(real_world_data_lab)
    
    # Update Current real world data's mean and std
    cur_mean = (1-gamma) * cur_mean + gamma * real_world_data_lab_mean
    cur_std = (1-gamma) * cur_std + gamma * real_world_data_lab_std
    new_std = cur_std/(sytheic_data_lab_std+1e-6)
    
    sytheic_data_lab = (sytheic_data_lab-sytheic_data_lab_mean) * new_std + cur_mean
    # convert back to RGB
    sytheic_data_rgb = lab_to_rgb(sytheic_data_lab)
    
    return sytheic_data_rgb, cur_mean,cur_std


# imagenet normalization
def ImageNetNormalization(data):
    imagenet_mean = torch.Tensor(IMAGENET_MEAN).type_as(data).view(1,3,1,1)
    imagenet_mean.requires_grad = False
    imagenet_val = torch.Tensor(IMAGENET_STD).type_as(data).view(1,3,1,1)
    imagenet_val.requires_grad = False
    
    data = (data - imagenet_mean)/imagenet_val

    return data