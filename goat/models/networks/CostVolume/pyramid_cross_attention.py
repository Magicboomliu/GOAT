import torch.nn as nn
import torch
import torch.nn.functional as F

def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x

def build_local_cross_attention(cross_attention,cur_disp,searching_radius,sample_nums):
    # cross attention is [B,H,W,W] : Before Softmax
    # searching radius
    # current dispairty: [B,1,H,W]
    # output should be [B,2*sample_nums+1,H,W] and its corresponding disparity
    
    b, h, w, w1 = cross_attention.shape #[B,H,W,W]
    cur_disp = cur_disp.permute(0,2,3,1) #[B,H,W,1]
    scale = w//w1
    x_range = torch.arange(0,w).view(1,1,w).expand(1,h,w).type_as(cur_disp)*1.0/scale #[1,H,W]
    standard_sample = x_range.permute(1,2,0).unsqueeze(0).repeat(b,1,1,1).float()  #[B,H,W,1]
    
    # Sample Center Point
    sample_center_point_old = (standard_sample - cur_disp)
    sample_center_point = torch.clamp(sample_center_point_old,min=0.0,max=w1-1) #[B,H,W,1]
    
    
    ########## Left Shift sample on attention map[B,H,W,W]: Disparity become bigger #############
    lower_bound = sample_center_point - searching_radius # FIXME Maybe smaller than Zero #[B,H,W,1]
    left_disp_intervals = searching_radius/(sample_nums-1) # Sampling intervals
    left_addition_index = torch.arange(sample_nums).type_as(cur_disp)
    left_addition_index = left_addition_index.view(1,1,1,sample_nums)
    left_sample_accumulation = left_addition_index * left_disp_intervals
    left_sample_candidates = lower_bound + left_sample_accumulation #[B,H,W,sample_nums]
    

    left_valid_mask = (left_sample_candidates>0).float()
    left_sample_candidates = torch.clamp(left_sample_candidates,min=0.0)
   
    left_sample_candidates_ceil = ste_ceil(left_sample_candidates)
    left_sample_candidates_floor = ste_floor(left_sample_candidates)
    left_sample_candidates_ceil = torch.clamp(left_sample_candidates_ceil,min=0,max=w1-1)
    left_sample_candidates_floor = torch.clamp(left_sample_candidates_floor,min=0,max=w1-1)
    
    
    left_floor_rate = left_sample_candidates_ceil - left_sample_candidates #[B,H,W,5]
    left_ceil_rate = 1.0 - left_floor_rate
    
    # print(left_sample_candidates_ceil)
    searching_range_left = cross_attention.shape[-1]
    left_sample_candidates_ceil = left_sample_candidates_ceil.long()
    left_sample_candidates_floor = left_sample_candidates_floor.long()
    left_sample_candidates_ceil = torch.clamp(left_sample_candidates_ceil,min=0,max=searching_range_left-1)
    left_sample_candidates_floor = torch.clamp(left_sample_candidates_floor,min=0,max=searching_range_left-1)
    left_ceil_volume = torch.gather(cross_attention,dim=-1,index=left_sample_candidates_ceil)
    left_floor_volume = torch.gather(cross_attention,dim=-1,index=left_sample_candidates_floor)


    
    left_final_volume = left_ceil_volume * left_ceil_rate +left_floor_volume * left_floor_rate
    left_final_volume = left_final_volume * left_valid_mask #[B,H,W,5]
    
    ########## Right Shift sample on attention map[B,H,W,W]: Disparity become bigger #############
    
    upper_bound = sample_center_point + searching_radius # FIXME Maybe smaller than Zero #[B,H,W,1]
    right_disp_intervals = searching_radius/(sample_nums-1) # Sampling intervals
    right_addition_index = torch.arange(sample_nums).type_as(cross_attention).float()
    
    right_addition_index = right_addition_index.view(1,1,1,sample_nums)
    right_sample_accumulation = right_addition_index * right_disp_intervals
    right_sample_candidates = upper_bound - right_sample_accumulation #[B,H,W,sample_nums]: from higher to lower
    

    
    right_disp_new = standard_sample - right_sample_candidates
    right_valid_mask1 = right_disp_new>0
    right_valid_mask2 = right_disp_new<w1-1
    right_valid_mask = right_valid_mask1 * right_valid_mask2
    right_valid_mask = right_valid_mask.bool()

    right_sample_candidates= torch.where(right_valid_mask,
                right_sample_candidates,sample_center_point_old)
    

    right_sample_candidates_ceil = ste_ceil(right_sample_candidates)
    right_sample_candiates_floor = ste_floor(right_sample_candidates)
    
    right_sample_candidates_ceil = torch.clamp(right_sample_candidates_ceil,min=0,max=w1-1)
    right_sample_candiates_floor = torch.clamp(right_sample_candiates_floor,min=0,max=w1-1)
    
    right_floor_rate = right_sample_candidates_ceil - right_sample_candidates #[B,H,W,5]
    right_ceil_rate = 1.0 - right_floor_rate

    searching_range_right = cross_attention.shape[-1]
    right_sample_candidates_ceil = right_sample_candidates_ceil.long()
    right_sample_candiates_floor = right_sample_candiates_floor.long()
    right_sample_candidates_ceil = torch.clamp(right_sample_candidates_ceil,min=0,max=searching_range_right-1)
    right_sample_candiates_floor = torch.clamp(right_sample_candiates_floor,min=0,max=searching_range_right-1)

    right_ceil_volume = torch.gather(cross_attention,dim=-1,index=right_sample_candidates_ceil)
    right_floor_volume = torch.gather(cross_attention,dim=-1,index=right_sample_candiates_floor)

    # right_ceil_volume  = torch.clamp(right_ceil_volume ,min=0,max=w-1)
    # right_floor_volume = torch.clamp(right_floor_volume,min=0,max=w-1)
    
    right_final_volume = right_ceil_volume * right_ceil_rate +right_floor_volume * right_floor_rate
    right_final_volume = right_final_volume * right_valid_mask.float() #[B,H,W,5]
    
    right_final_volume = torch.flip(right_final_volume,dims=[-1])[:,:,:,1:]
    
    # Final Cost Volume
    final_local_cost_volume = torch.cat((left_final_volume,right_final_volume),dim=-1)
    
    # correspinding disparities
    disp_candidates = torch.arange(-(sample_nums-1),(sample_nums)).to(final_local_cost_volume.device) * left_disp_intervals + cur_disp

    # disp_candidates = torch.zeros_like(cur_disp)
    return final_local_cost_volume, disp_candidates

def regress_disp(cost_volume):
    '''
    :param att:         B * H * W * W
    :param valid_mask:  B * 1 * H * W
    '''
    cost_volume = torch.tril(cost_volume)
    cost_volume = torch.exp(cost_volume - cost_volume.max(-1)[0].unsqueeze(-1))
    cost_volume = torch.tril(cost_volume)
    att = cost_volume / (cost_volume.sum(-1, keepdim=True) + 1e-8)
    
    b, h, w, _ = att.shape
    index = torch.arange(w).view(1, 1, 1, w).to(att.device).float()    # index: 1*1*1*w
    disp_ini = index - torch.sum(att * index, dim=-1).view(b, 1, h, w)

    return disp_ini

def regress_disp_2(cost_volume):
    '''
    :param att:         B * H * W * W
    :param valid_mask:  B * 1 * H * W
    '''
    cost_volume = torch.tril(cost_volume)

    zero_mask = cost_volume<=0
    cost_volume = zero_mask.float()*-10 + cost_volume
    cost_volume = torch.exp(cost_volume)
    cost_volume = torch.tril(cost_volume)
    att = cost_volume / (cost_volume.sum(-1, keepdim=True) + 1e-8)
    
    b, h, w, _ = att.shape
    index = torch.arange(w).view(1, 1, 1, w).to(att.device).float()    # index: 1*1*1*w
    disp_ini = index - torch.sum(att * index, dim=-1).view(b, 1, h, w)

    return disp_ini



class PyrmaidCostVolume(nn.Module):
    def __init__(self,radius,nums_levels,
                 sample_points):
        super(PyrmaidCostVolume,self).__init__()
        self.radius = radius
        self.nums_levels = nums_levels
        self.sample_points = sample_points


    def forward(self,cross_attention,cur_disp):
        # get the Cross Attention
        cross_attention_pyramid = []
        cross_attention_pyramid.append(cross_attention)
        for i in range(self.nums_levels-1):
            B,H,W1,W2 = cross_attention.shape
            cross_attention = cross_attention.view(B,-1,W2)
            cross_attention = F.avg_pool1d(cross_attention,2,stride=2)
            cross_attention= cross_attention.contiguous().view(B,H,W1,W2//2)
            cross_attention_pyramid.append(cross_attention)
        
        # Index the Cross Attention
        out_pyramid =[]
        out_disp_pyramid=[]
        for i in range(self.nums_levels):
            corr = cross_attention_pyramid[i]
            ref_disp = cur_disp*1.0 /(2**i)
            local_cost_volume,local_disp= build_local_cross_attention(corr,ref_disp,self.radius,self.sample_points)
    
            out_pyramid.append(local_cost_volume.permute(0,3,1,2))
            out_disp_pyramid.append(local_disp.permute(0,3,1,2))
            
        out_cross_attention = torch.cat(out_pyramid,dim=1)
        out_disp = torch.cat(out_disp_pyramid,dim=1)
        
        return out_cross_attention,out_disp
        

if __name__=="__main__":
        
    cross_attention_example = torch.abs(torch.randn(80*160*160).view(1,80,160,160)).float()
    cross_attention_example = torch.tril(cross_attention_example).cuda()

    cur_disp_example = torch.ones(1,1,80,160).to(cross_attention_example.device).cuda()

    
    pyca = PyrmaidCostVolume(radius=3,nums_levels=4,sample_points=4).cuda()
    out_cross_attention,out_disp = pyca(cross_attention_example,cur_disp_example)
    
    
    print(out_cross_attention.shape)
    # print("----------------------")
    # print(out_disp.shape)