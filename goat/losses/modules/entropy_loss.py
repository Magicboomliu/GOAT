import torch
import torch.nn as nn
import torch.nn.functional as F

def torch_1d_sample(source, sample_points, mode='linear'):
    """
    linearly sample source tensor along the last dimension
    input:
        source [N,D1,D2,D3...,Dn]
        sample_points [N,D1,D2,....,Dn-1,1]
    output:
        [N,D1,D2...,Dn-1]
    """
    idx_l = torch.floor(sample_points).long().clamp(0, source.size(-1) - 1)
    idx_r = torch.ceil(sample_points).long().clamp(0, source.size(-1) - 1)

    if mode == 'linear':
        weight_r = sample_points - idx_l
        weight_l = 1 - weight_r
    elif mode == 'sum':
        weight_r = (idx_r != idx_l).int()  # we only sum places of non-integer locations
        weight_l = 1
    else:
        raise Exception('mode not recognized')

    out = torch.gather(source, -1, idx_l) * weight_l + torch.gather(source, -1, idx_r) * weight_r
    return out.squeeze(-1)



def compute_gt_location(scale: float, attn_weight, disp):
    """
    Find target locations using ground truth disparity.
    Find ground truth response at those locations using attention weight.

    :param scale: high-res to low-res disparity scale
    :param attn_weight: attention weight (output from _optimal_transport), [N,H,W,W]
    :param disp: ground truth disparity
    :return: response at ground truth location [N,H,W,1] and target ground truth locations [N,H,W,1]
    """
    # compute target location at full res
    _, _, w = disp.size()
    pos_l = torch.linspace(0, w - 1, w)[None,].to(disp.device)  # 1 x 1 x W (left)
    target = (pos_l - disp)[:, ::4, ::4, None]  # N x H x W (left) x 1
    target = target / scale  # scale target location

    # compute ground truth response location for rr loss
    gt_response = torch_1d_sample(attn_weight, target, 'linear')  # NxHxW_left

    return gt_response, target

def get_occlusion_response(attn_ot,occlusion_mask):
    occlusion_mask = occlusion_mask[:, ::4, ::4]
    return attn_ot[...,:,-1][occlusion_mask]


def compute_rr_loss(attn_ot,target_disp,target_occlusion):
    """
    compute rr loss
    
    :param outputs: dictionary, outputs from the network
    :param inputs: input data
    :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
    :return: rr loss
    """""
    invalid_mask = (target_disp<0) & (target_disp>192)
    invalid_mask = invalid_mask + target_occlusion
    invalid_mask = invalid_mask.bool()
    if invalid_mask is not None:
        invalid_mask = invalid_mask[:, ::4, ::4]

    # compute rr loss in non-occluded region
    gt_response, target = compute_gt_location(scale=4,attn_weight=attn_ot[...,:,:-1],disp=target_disp)
    eps = 1e-6
    rr_loss = - torch.log(gt_response + eps)

    if invalid_mask is not None:
        rr_loss = rr_loss[~invalid_mask]

    # if there is occlusion
    try:
        occlusion_response = get_occlusion_response(attn_ot,occlusion_mask=target_occlusion)
        rr_loss_occ_left = - torch.log(occlusion_response + eps)
        # print(rr_loss_occ_left.shape)
        rr_loss = torch.cat([rr_loss, rr_loss_occ_left])
    except KeyError:
        pass
    return rr_loss.mean()


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

    invalid_mask = (target_disp<0) & (target_disp>192)
    invalid_mask = invalid_mask + occ_gt
    invalid_mask = invalid_mask.bool()
    
    
    entropy_loss_occ = -torch.log(occ_pred[occ_mask] + eps)
    entropy_loss_noc = - torch.log(
        1.0 - occ_pred[~invalid_mask] + eps)  # invalid mask includes both occ and invalid points

    entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])

    return entropy_loss.mean()