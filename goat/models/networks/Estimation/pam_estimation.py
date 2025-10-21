import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def regress_disp(att):    
    b, h, w, _ = att.shape
    
    # att_vis = att.squeeze(0)
    # a = att_vis[40,30].cpu().numpy()
    # mean = torch.mean(att_vis,dim=-1,keepdim=True)
    # std = torch.sqrt(torch.sum(torch.square(att_vis-mean),dim=-1,keepdim=True)/w)
    # print(std.mean())
    # plt.plot(a)
    # plt.show()
    # quit()
    
    index = torch.arange(w).view(1, 1, 1, w).to(att.device).float()    # index: 1*1*1*w
    disp = index - torch.sum(att * index, dim=-1).view(b, 1, h, w)
    
    return disp


class DisparityOcclusionRegression(nn.Module):
    def __init__(self,feature_dim=16):
        super(DisparityOcclusionRegression,self).__init__()
        
        self.occluded_regress = nn.Sequential(
            weight_norm(nn.Conv2d(1 + 3, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self,cross_attention,left_img):
        cross_attention_left2right, cross_attention_right2left = cross_attention
        b,h,w,_ = cross_attention_left2right.shape
        
        h_origin = left_img.shape[2]
        scale = h_origin //h
        left_img = F.interpolate(left_img,scale_factor=1./scale,mode='bilinear',align_corners=False)
        
        # Change it to tril matrix
        cross_attention_left2right = torch.tril(cross_attention_left2right) # Downer Traniagle
        left_zero_mask = cross_attention_left2right<=0
        cross_attention_left2right = left_zero_mask.float()*-10.0 + cross_attention_left2right
        cross_attention_left2right = torch.clamp(cross_attention_left2right,min=-15,max=15)
        cross_attention_left2right = torch.exp(cross_attention_left2right)
        cross_attention_left2right = torch.tril(cross_attention_left2right) # Cross attention make sense.
        att_left2right = cross_attention_left2right/(cross_attention_left2right.sum(-1,keepdim=True) +1e-8)
        
        
        cross_attention_right2left = torch.triu(cross_attention_right2left) # Upper traniagle
        right_zero_mask = cross_attention_right2left<=0
        cross_attention_right2left = right_zero_mask.float()*-10.0 + cross_attention_right2left
        cross_attention_right2left = torch.clamp(cross_attention_right2left,min=-15,max=15)
        cross_attention_right2left = torch.exp(cross_attention_right2left)
        cross_attention_right2left = torch.triu(cross_attention_right2left) # Cross Attention Make Sense.
        att_right2left = cross_attention_right2left/(cross_attention_right2left.sum(-1,keepdim=True)+1e-8)
        
        # Occlusion Mask
        valid_mask_left = torch.sum(att_right2left,-2).unsqueeze(1)
        occlusion_feature = torch.cat((valid_mask_left,left_img),dim=1)
        occlusion = self.occluded_regress(occlusion_feature)
        
        # Disp 
        disp = regress_disp(att_left2right)
        
        return disp,occlusion
        
        
        
        