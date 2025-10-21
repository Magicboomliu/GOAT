import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x


def build_local_cost_volume_fixed(cost_volume,cur_disp,searching_radius,sample_nums):
    
    # Cost Volume Shape
    B,D,H,W = cost_volume.shape
    
    # Get sample candidates
    lower_bound = cur_disp - searching_radius
    upper_bound = cur_disp + searching_radius
    sample_intervals = (upper_bound-lower_bound) *1.0/(sample_nums)    
    addition_summation = torch.arange(sample_nums+1).type_as(cur_disp)
    addition_summation=addition_summation.view(1,sample_nums+1,1,1)
    sampling_candiate_intervals = addition_summation * sample_intervals
    sampling_candidates =lower_bound + sampling_candiate_intervals
    
    # valid mask
    sample_candidate_ceil = ste_ceil(sampling_candidates)
    sample_candidate_floor = ste_floor(sampling_candidates)
    
    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=D-1)
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=D-1)
    
    # Linear interplotation
    floor_rate =(sample_candidate_ceil- sampling_candidates)
    ceil_rate = 1.0 - floor_rate
    
    ceil_volume = torch.gather(cost_volume,dim=1,index=sample_candidate_ceil.long())
    floor_volume = torch.gather(cost_volume,dim=1,index=sample_candidate_floor.long())
    
    final_volume = ceil_volume*ceil_rate+ floor_volume*floor_rate
    
    return final_volume


class PyrmaidCostVolume(nn.Module):
    def __init__(self,radius,nums_levels,
                 sample_points):
        super(PyrmaidCostVolume,self).__init__()
        self.radius = radius
        self.nums_levels = nums_levels
        self.sample_points = sample_points
        
    
    def forward(self,cost_volume,radius,cur_disp):
        
        # Get the Cost Volume.
        cost_volume_pyramid = []
        cost_volume_pyramid.append(cost_volume)
        # from full searching range to 1/2 searching range.
        for i in range(self.nums_levels-1):
            B,D,H,W = cost_volume.shape
            cost_volume = cost_volume.view(B,D,-1).permute(0,2,1)
            cost_volume = F.avg_pool1d(cost_volume,2,stride=2)
            cost_volume = cost_volume.permute(0,2,1).contiguous().view(B,D//2,H,W)
            cost_volume_pyramid.append(cost_volume)
        
        # Index the Cost Volume.
        
        out_pyramid = []
        for i in range(self.nums_levels):
            corr = cost_volume_pyramid[i]
            ref_disp = cur_disp*1.0 /(2**i)
            local_cost_volume = build_local_cost_volume_fixed(corr,ref_disp,radius,self.sample_points)
            out_pyramid.append(local_cost_volume)
        
        out = torch.cat(out_pyramid,dim=1)
    
        return out