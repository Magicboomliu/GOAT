import torch
import torch.nn as nn
import torch.nn.functional as F
from goat.losses.modules.entropy_loss import compute_entropy_loss

def disparityLoss_occlusionLoss(pred_disp,pred_occ,gt_disp,gt_occlusion):
    weight=[0.8,1.0]
    valid_mask = (gt_disp>0) & (gt_disp<192)
    
    disp_loss = 0
    if isinstance(pred_disp,list) or isinstance(pred_disp,tuple):
        for idx, pred_d in enumerate(pred_disp):
            disp_loss_cur =  F.smooth_l1_loss(input=pred_d[valid_mask],target=gt_disp[valid_mask],size_average=True)
            disp_loss += disp_loss_cur * weight[idx]
    else:
        disp_loss =  F.smooth_l1_loss(input=pred_disp[valid_mask],target=gt_disp[valid_mask],size_average=True)

    
    occ_loss = compute_entropy_loss(occ_pred=pred_occ,occ_gt=gt_occlusion,target_disp=gt_disp)
    
    loss = disp_loss + occ_loss
    
    return loss,disp_loss,occ_loss