import torch.nn as nn
import torch
import torch.nn.functional as F

def compute_entropy_loss(occ_pred, occ_gt, target_disp):
    """
    compute binary entropy loss on occlusion mask

    :param occ_pred: occlusion prediction, [N,H,W]
    :param inputs: input data
    :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
    :return: binary entropy loss
    """
    eps = 1e-6

    occ_mask = occ_gt.bool()

    invalid_mask = occ_gt
    invalid_mask = invalid_mask.bool()
    
    
    entropy_loss_occ = -torch.log(occ_pred[occ_mask] + eps)
    entropy_loss_noc = - torch.log(
        1.0 - occ_pred[~invalid_mask] + eps)  # invalid mask includes both occ and invalid points

    entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])

    return entropy_loss.mean()



def fusion_loss(occ_pred, occ_gt, target_disp):
    
    entropy_loss = compute_entropy_loss(occ_pred=occ_pred,occ_gt=occ_gt,target_disp=target_disp)
    
    l1_loss = F.smooth_l1_loss(occ_pred.float(),occ_gt.float(),reduction='mean')
    
    return entropy_loss + l1_loss
    
