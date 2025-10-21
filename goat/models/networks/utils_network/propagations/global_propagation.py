import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGlobalProapagation(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(FeatureGlobalProapagation,self).__init__()
        # Q and K 
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                source,
                target,
                **kwargs):
        
        b, c, h, w = source.size()
        query = source.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]
        
        value = target.view(b, target.size(1), h * w).permute(0, 2, 1)  # [B, H*W, C2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)
    

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # FIXME: Not sure whether it will work or not.
        out = out + target
        
        return out,prob




class FeatureDispAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(FeatureDispAttention,self).__init__()
        # Q and K 
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_local_window_attn(self, feature0, disp,
                                  local_window_radius=1,
                                  ):
        assert disp.size(1) == 1
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(disp, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 1*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, 1, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, 1)  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, 1).permute(0, 3, 1, 2).contiguous()  # [B, 1, H, W]

        return out

    def forward(self,
                feature0,
                disp,local_window_attn=False,
                local_window_radius=1,
                **kwargs):

        # q, k: feature [B, C, H, W], v: disp [B, 1, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, disp,
                                                  local_window_radius=local_window_radius)
        
        b, c, h, w = feature0.size()
        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]
        
        value = disp.view(b, disp.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]
     
        return out


if __name__=="__main__":
    guidance_feature = torch.randn(1,128,40,80).cuda()
    
    target_feature = torch.randn(1,10,40,80).cuda()
    
    
    feature_global_propagation = FeatureGlobalProapagation(in_channels=128).cuda()
    
    out = feature_global_propagation(guidance_feature,target_feature)
    
    print(out.shape)