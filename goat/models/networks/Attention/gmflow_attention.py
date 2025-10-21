import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureStereoAggrgegation(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 prop_channels=256,
                 **kwargs,
                 ):
        super(FeatureStereoAggrgegation, self).__init__()

        self.in_channels = in_channels
        self.prop_channels = prop_channels
        
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                context_feature, 
                occlusion_mask,
                aggreagated_feat,
                local_window_radius=1,
                **kwargs,
                ):
        
        guidance_feat = context_feature
        
        
        # occlusion mask and non-occluded mask
        occlusion_mask = occlusion_mask.float()
        inverse_occlusion_mask = torch.ones_like(occlusion_mask) - occlusion_mask
        occlusion_mask = occlusion_mask.detach()
        inverse_occlusion_mask = inverse_occlusion_mask.detach()
        
        # local feature using as the non-occluded regions   
        aggreagated_feat_local = self.forward_local_window_attn(guidance_feat,aggreagated_feat,
                                                          local_window_radius=local_window_radius)
        
        
        b, c, h, w = context_feature.size()
        query = context_feature.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = aggreagated_feat.view(b, aggreagated_feat.size(1), h * w).permute(0, 2, 1)  # [B, H*W, C]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, C]
        
        # global feature using as the occluded regions
        aggreagated_feat_global = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        
        out = aggreagated_feat_global * occlusion_mask + aggreagated_feat_local * inverse_occlusion_mask
        

        return out

    def forward_local_window_attn(self, context_feature, 
                                  aggregated_feature,
                                  local_window_radius=1,
                                  ):
    
        assert local_window_radius > 0

        # get the context feature shape.
        b, c, h, w = context_feature.size()        
        
        # get the current feature
        feature0_reshape = self.q_proj(context_feature.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]
        # kernel size
        kernel_size = 2 * local_window_radius + 1
        feature0_proj = self.k_proj(context_feature.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)


        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]
        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]
        

        # aggregated feature
        feat_window = F.unfold(aggregated_feature, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]
        feat_window = feat_window.view(b, self.prop_channels, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, self.prop_channels)  # [B*H*W, (2R+1)^2, 2]
        
        # score softmax
        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]
        
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, feat_window).view(b, h, w, self.prop_channels).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out


class FeatureStereoAggrgegationV2(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 prop_channels=256,
                 **kwargs,
                 ):
        super(FeatureStereoAggrgegationV2, self).__init__()

        self.in_channels = in_channels
        self.prop_channels = prop_channels
        
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                context_feature, 
                occlusion_mask,
                aggreagated_feat,
                **kwargs,
                ):
        
        guidance_feat = context_feature
        local_window_radius = 1
        
        
        # occlusion mask and non-occluded mask
        occlusion_mask = occlusion_mask.type_as(guidance_feat)
        inverse_occlusion_mask = torch.ones_like(occlusion_mask) - occlusion_mask
        occlusion_mask = occlusion_mask.detach()
        inverse_occlusion_mask = inverse_occlusion_mask.detach()
        
        # local feature using as the non-occluded regions   
        aggreagated_feat_local = self.forward_local_window_attn(guidance_feat,aggreagated_feat,
                                                          local_window_radius=local_window_radius)
        
        
        b, c, h, w = context_feature.size()
        query = context_feature.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = aggreagated_feat.view(b, aggreagated_feat.size(1), h * w).permute(0, 2, 1)  # [B, H*W, C]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, C]
        
        # global feature using as the occluded regions
        aggreagated_feat_global = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        
        out = aggreagated_feat_global * occlusion_mask + aggreagated_feat_local * inverse_occlusion_mask
        

        return out

    def forward_local_window_attn(self, context_feature, 
                                  aggregated_feature,
                                  local_window_radius=1,
                                  ):
    
        assert local_window_radius > 0

        # get the context feature shape.
        b, c, h, w = context_feature.size()        
        # get the current feature
        feature0_reshape = self.q_proj(context_feature.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]
        # kernel size
        kernel_size = 2 * local_window_radius + 1
        feature0_proj = self.k_proj(context_feature.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)


        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]
        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]
        

        # aggregated feature
        feat_window = F.unfold(aggregated_feature, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]
        feat_window = feat_window.view(b, self.prop_channels, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, self.prop_channels)  # [B*H*W, (2R+1)^2, 2]
        
        # score softmax
        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]
        
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, feat_window).view(b, h, w, self.prop_channels).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out









if __name__=="__main__":
    
    guidance_feature = torch.randn(1,128,40,80).cuda()

    aggregated_feature = torch.randn(1,128,40,80).cuda()
    
    occlusion_mask = torch.sigmoid(torch.randn(1,1,40,80)).cuda()
    
    
    feature_aggregatetion = FeatureStereoAggrgegation(in_channels=128).cuda()
    
    feat_out = feature_aggregatetion(guidance_feature,occlusion_mask,aggregated_feature,local_window_radius=1)
    
    print(feat_out.shape)
    
    pass