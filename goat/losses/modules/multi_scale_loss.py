import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")



def EPE_Loss(disp_infer,disp_gt):
    mask = (disp_gt>0) & (disp_gt<192)
    disp_infer = disp_infer[mask]
    disp_gt = disp_gt[mask]
    return F.l1_loss(disp_infer,disp_gt,size_average=True)


# "Multi-Scale Loss"
class MultiScaleLoss(nn.Module):
    def __init__(self, scales, weights=None, 
                 loss='Smooth_l1',
                 downsample=1):
        super(MultiScaleLoss, self).__init__()
        self.weights = weights
        self.downsample = downsample
        assert(len(self.weights) == scales)         
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)

        if self.loss=='Smooth_l1':
            self.loss = self.smoothl1
    
    def forward(self, disp_infer, disp_gt):
        loss = 0
        
        # Training
        if (type(disp_infer) is tuple) or (type(disp_infer) is list):
            for i, input_ in enumerate(disp_infer):
                assert input_.size(-2)==disp_gt.size(-2)
                target = disp_gt
                mask = (target<192) & (target>0)
                mask.detach_()
                input_ = input_[mask]
                target_ = target[mask]
                loss+= self.smoothl1(input_,target_) * self.weights[i]
        
        # Validation
        else:
            mask = (disp_gt<192) & (disp_gt>0)
            mask = mask.detach_()
            
            loss = self.loss(disp_infer[mask],disp_gt[mask])
        
        return loss
                

def multiscaleloss(scales=5, downscale=4, weights=None, loss='Smooth_l1'):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales, weights, loss,downscale)