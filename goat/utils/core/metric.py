import torch
import torch.nn.functional as F
import numpy as np

def D1_metric(D_pred, D_gt):
    E = torch.abs(D_pred - D_gt)
    E_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(E_mask.float())

def P1_metric(D_pred, D_gt):
    E = torch.abs(D_pred - D_gt)
    E_mask = (E > 1)
    return torch.mean(E_mask.float())


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def Disparity_EPE_Loss(predicted_disparity,gt_disparity):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<192
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6

    epe_val = torch.abs(predicted_disparity*valid_mask-gt_disparity*valid_mask).sum()/(valid_mask.sum()+eps)
    return epe_val

def Disparity_EPE_Loss_KITTI(predicted_disparity,gt_disparity):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<320
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6
    epe_val = torch.abs(predicted_disparity*valid_mask-gt_disparity*valid_mask).sum()/(valid_mask.sum()+eps)
    return epe_val




def P1_Value(predicted_disparity,gt_disparity):
    valid_mask = gt_disparity>0
    eps =1e-6
    E = torch.abs(predicted_disparity*valid_mask- gt_disparity*valid_mask)
    E_mask = (E>1)
    
    return E_mask.sum()*1.0/(valid_mask.sum()+eps)


def D1_metric(predicted_disparity,gt_disparity):
    valid_mask = gt_disparity>0
    eps =1e-6
    E = torch.abs(predicted_disparity*valid_mask- gt_disparity*valid_mask)
    E_mask = (E > 3) & (E / (gt_disparity*valid_mask).abs() > 0.05)
    
    return E_mask.sum()*1.0/(valid_mask.sum()+eps)




@torch.no_grad()
def Occlusion_EPE(predicted_occlusion,target_occlusion,disp_gt):
    mask = (disp_gt>0) & (disp_gt<192)
    predicted_occlusion = predicted_occlusion.float()[mask]
    target_occlusion = target_occlusion.float()[mask]
    return F.l1_loss(predicted_occlusion,target_occlusion,size_average=True)


@torch.no_grad()
def Occlusion_EPE_KITTI(predicted_occlusion,target_occlusion,disp_gt):
    mask = (disp_gt>0) & (disp_gt<320)
    predicted_occlusion = predicted_occlusion.float()[mask]
    target_occlusion = target_occlusion.float()[mask]
    return F.l1_loss(predicted_occlusion,target_occlusion,size_average=True)



@torch.no_grad()
def compute_iou(pred, occ_mask, target_disp):
    """
    compute IOU on occlusion
    :param pred: occlusion prediction [N,H,W]
    :param occ_mask: ground truth occlusion mask [N,H,W]
    :param loss_dict: dictionary of losses
    :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
    """
    invalid_mask = (target_disp<0) & (target_disp>192)
    invalid_mask = invalid_mask + occ_mask
    invalid_mask = invalid_mask.bool()
    
    # threshold
    pred_mask = pred > 0.5
    # iou for occluded region
    inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
    union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask).sum()
    # iou for non-occluded region
    inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
    union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask), ~invalid_mask).sum()
    # aggregate
    iou = (inter_occ + inter_noc).float() / (union_occ + union_noc)
    return iou

