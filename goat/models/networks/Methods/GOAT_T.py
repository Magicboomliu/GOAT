import torch
import torch.nn as nn
import torch.nn.functional as F
from goat.models.networks.Backnbones.raft_extractor import BasicEncoder
from goat.models.networks.Attention.global_attention import AttentionRelative
from goat.models.networks.Attention.PAMBlock import PAM_stageV2
from goat.models.networks.Estimation.pam_estimation import DisparityOcclusionRegression
from goat.models.networks.CostVolume.pyramid_cross_attention import PyrmaidCostVolume
from goat.models.networks.Aggregation.transformer import SingleFeatureTransformer, FeatureFlowAttention, FeatureTransformer
from goat.models.networks.Aggregation.transformer_utils import feature_add_position_single, feature_add_position
from goat.models.networks.Update.OGAUpdate import OcclusionAwarenessGlobalAggregation

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

class GOAT_T(nn.Module):
    def __init__(self,
                 radius = 2,
                 num_levels =3,
                 sample_points =3,
                 dropout=0.,
                 refine_type='g',
                 up_scale=3):
        super().__init__()
        
        self.refine_type = refine_type
        self.up_scale = up_scale
        # HyperParameters
        cdim = 128
        self.hidden_dim  = 128
        self.context_dim = 128
        self.radius = radius
        self.num_levels = num_levels
        self.sample_points = sample_points
        
        # Feature NetWork, Context Network, and update Blocks
        self.fnet = BasicEncoder(output_dim=256,norm_fn='instance',dropout=dropout)
        self.cnet = BasicEncoder(output_dim=256,norm_fn='batch',dropout=dropout)
        # feature aggregation
        self.feature_aggregation = FeatureTransformer(num_layers=4,d_model=256,
                                                      nhead=1,attention_type='swin',
                                                      ffn_dim_expansion=4)
        # Context Attention Here
        self.att =AttentionRelative(args='position_and_content', 
                             dim=cdim, 
                             heads=1, 
                             max_pos_size=160, dim_head=cdim)

        # PAM Stage
        self.cross_attention = PAM_stageV2(channels=256)
        # Disparity and Initial Occlusion Estimation Network.
        self.disp_occlusion = DisparityOcclusionRegression()
        # Pyramid Cross Attention
        self.pyca = PyrmaidCostVolume(radius=self.radius,nums_levels=self.num_levels,
                                              sample_points=self.sample_points)
        # Update module
        if self.refine_type=='g':
            self.update_block = OcclusionAwarenessGlobalAggregation(hidden_dim=128,
                                                                cost_volume_dimension=(2*self.sample_points-1) * self.num_levels)

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 **self.up_scale
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        
    
    def forward(self,left_image,right_image,
                      iters=12,upsample=True,test_mode=False):
        
        left_image = left_image.contiguous()
        right_image = right_image.contiguous()
        
        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # Run the feature network
        with autocast(enabled=False):
            fmap1, fmap2 = self.fnet([left_image, right_image])

        fmap1 = fmap1.float() # [B,256,H//8,W//8] : exist negative numbers
        fmap2 = fmap2.float()
        # perform the feature aggregation
        fmap1,fmap2 = feature_add_position(fmap1,fmap2,attn_splits=2,feature_channels=256)
        # feature aggregation
        fmap1,fmap2 = self.feature_aggregation(fmap1,fmap2,attn_num_splits=2)
        b,_,h,w = fmap1.shape
        
        # Run the context network
        with autocast(enabled=False):
            cnet = self.cnet(left_image) # [B,256,H//8,W//8]
            hidden_state, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # Hidden State
            hidden_state = torch.tanh(hidden_state)
            # Context Feature
            inp = torch.relu(inp)
            attention = self.att(inp)
            attention = attention.float()
            hidden_state = hidden_state.float()
            inp = inp.float()
            cnet = cnet.float()

        # Cross Attention: 1/8 Stage
        cross_attention_s0 = [
            torch.zeros(b,h,w,w).to(left_image.device),
            torch.zeros(b,h,w,w).to(right_image.device)
        ]
        
        # Fmap1 and Fmap2
        fmap1, fmap2, cross_attention_s1 = self.cross_attention(fmap1,fmap2,cross_attention_s0)
        disp_init,occlusion_initial = self.disp_occlusion(cross_attention_s1,left_image)
        disparity_pyramid = []
        
        disp3 = F.interpolate(disp_init,scale_factor=8.0,mode='bilinear',align_corners=False) *8.0
        disparity_pyramid.append(disp3)
        cur_disp = disp_init
        ref_coords = torch.zeros_like(disp_init).type_as(fmap1)
        

        for itr in range(iters):
            cur_disp = cur_disp.detach()
            corr,corr_disp = self.pyca(cross_attention_s1[0],cur_disp)
            disp= cur_disp - ref_coords
            with autocast(enabled=False):
                hidden_state,up_mask,delta_disp = self.update_block(hidden_state,inp,corr,disp,occlusion_initial,
                                                                    corr_disp,attention)
            # Update the current disparity
            cur_disp = F.relu(cur_disp + delta_disp,True)
            # Upsample
            disp_up = self.upsample_flow(cur_disp,up_mask)
            disparity_pyramid.append(disp_up)
            
        
        # Upsample to higher resolution
        occlusion_final = F.interpolate(occlusion_initial,scale_factor=8.0,mode='nearest')

        if test_mode:
            return cur_disp,disp_up,occlusion_final

        return disparity_pyramid,occlusion_final



if __name__=='__main__':
    
    left_input = torch.randn(1,3,320,640).cuda()
    right_input = torch.randn(1,3,320,640).cuda()
    
    goat = GOAT_T(radius=3,num_levels=4,sample_points=4,dropout=0,refine_type='g',up_scale=3).cuda()
    
    
    with torch.no_grad():
        outputs,pred_occlusion =  goat(left_input, right_input,iters=12,test_mode=False)        
        for out in outputs:
            print(out.shape)

