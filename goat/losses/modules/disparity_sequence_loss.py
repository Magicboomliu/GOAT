import torch
import torch.nn.functional as F
import torch.nn as nn
from goat.losses.modules.entropy_loss import compute_entropy_loss
from goat.utils.core.disparity_warper import disp_warp


# Disparity sequence loss with last scale occlusion
def sequence_lossV2(pred_disp,gt_disp,
                    pred_occ ,gt_occ,
                    loss_gamma=0.9,max_disp=192):
    
    initial_disp =  pred_disp[0]
    pred_disp=pred_disp[1:]
    
    N_predictions = len(pred_disp)

    assert N_predictions >0
    
    disp_loss = 0.0
    occ_loss = 0.0
    
    valid_mask = (gt_disp>0)*(gt_disp<max_disp)
    valid_mask = valid_mask.type_as(gt_disp)
    
    initial_disp = torch.where(torch.isnan(initial_disp),torch.zeros_like(initial_disp),initial_disp)
    valid_mask = torch.where(torch.isnan(valid_mask),torch.zeros_like(valid_mask),valid_mask)
    gt_disp = torch.where(torch.isnan(gt_disp),torch.zeros_like(gt_disp),gt_disp)
    gt_disp = torch.where(torch.isnan(gt_disp),torch.zeros_like(gt_disp),gt_disp)

    initial_disp = torch.where(torch.isinf(initial_disp),torch.zeros_like(initial_disp),initial_disp)
    valid_mask = torch.where(torch.isinf(valid_mask),torch.zeros_like(valid_mask),valid_mask)
    gt_disp = torch.where(torch.isinf(gt_disp),torch.zeros_like(gt_disp),gt_disp)
    gt_disp = torch.where(torch.isinf(gt_disp),torch.zeros_like(gt_disp),gt_disp)
    
    #initial loss
    assert not torch.isnan(initial_disp).any() and not torch.isinf(initial_disp).any()
    assert not torch.isnan(valid_mask).any() and not torch.isinf(valid_mask).any()
    assert not torch.isnan(gt_disp).any() and not torch.isinf(gt_disp).any()
    
    
    initial_disp_loss = F.smooth_l1_loss(input=initial_disp*valid_mask,target=gt_disp*valid_mask,size_average=True)
    disp_loss += initial_disp_loss *0.4
    
    # sequence loss
    pred_disp = [p * valid_mask for p in pred_disp]
    
    for i in range(N_predictions):
        assert not torch.isnan(pred_disp[i]).any() and not torch.isinf(pred_disp[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(N_predictions - 1))
        i_weight = adjusted_loss_gamma**(N_predictions - i - 1)
        i_loss = (pred_disp[i] - gt_disp).abs()
        disp_loss += i_weight * i_loss[valid_mask.bool()].mean()
    
    occ_loss = compute_entropy_loss(occ_pred=pred_occ,occ_gt=gt_occ,target_disp=gt_occ)
    
    loss = disp_loss + occ_loss
    return loss, disp_loss,occ_loss

