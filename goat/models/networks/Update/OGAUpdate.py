import torch
import torch.nn as nn
import torch.nn.functional as F
from goat.models.networks.Attention.global_attention import Aggregate
from goat.models.networks.utils_network.propagations.local_propagation import LocalAffinityPropagation
from torch.utils.checkpoint import checkpoint


def create_custom_self_attn(module):
    def custom_self_attn(*inputs):
        return module(*inputs)
    return custom_self_attn



# Disparity residual prediction head.
class DisparityHead(nn.Module):
    def __init__(self,input_dim=128,hidden_dim=256):
        super(DisparityHead,self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim,hidden_dim,3,padding=1)
        self.conv2 = nn.Conv2d(hidden_dim,1,3,padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))
        
# ConvGRU
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        # hidden state.
        hx = torch.cat([h, x], dim=1)
        # remember gate.
        z = torch.sigmoid(self.convz(hx))
        # input gate
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

# PAM Depth Update Blocks
class PAMDepthEncoder(nn.Module):
    def __init__(self,cost_volume_dimension):
        super().__init__()
        self.cost_volime_dimension = cost_volume_dimension
        self.convc1 = nn.Conv2d(cost_volume_dimension, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)
        self.convc3 = nn.Conv2d(cost_volume_dimension,128,3,padding=1,stride=1)
        
        self.convf1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+128+128, 128-1, 3, padding=1)
    
    def forward(self,disp,corr,corr_disp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        
        disp_residual = corr_disp - disp
        disp_residual_feat = F.relu(self.convc3(disp_residual))
        
        disp_feat = F.relu(self.convf1(disp))
        disp_feat = F.relu(self.convf2(disp_feat))

        cor_disp_feat = torch.cat([cor, disp_residual_feat,disp_feat], dim=1)
        out = F.relu(self.conv(cor_disp_feat))
        return torch.cat([out, disp], dim=1)




class OcclusionAwarenessGlobalAggregation(nn.Module):
    def __init__(self,hidden_dim=128,
                 cost_volume_dimension=15):
        super(OcclusionAwarenessGlobalAggregation,self).__init__()
        
        # Cost Volune and depth aggregation
        self.encoder = PAMDepthEncoder(cost_volume_dimension=cost_volume_dimension)
        # GRU feature aggregation
        self.gru = SepConvGRU(hidden_dim=hidden_dim,input_dim=128+hidden_dim)
        # Global Aggregator
        self.global_aggregator = Aggregate(dim=256,heads=1,dim_head=256)
        # GRU feature aggreagtion
        self.gru = SepConvGRU(hidden_dim=hidden_dim,input_dim=128+hidden_dim)
        # Predict the disparity residual
        self.disp_head = DisparityHead(hidden_dim,hidden_dim=256)
        
        # Predict the upsample mask.
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        
    
    def forward(self,hidden_state,
                cnn_input,
                corr,
                disp,
                occ,
                disp_residual,
                attention):
        
        
        # Encode the current disparity, current corr and current disparity residual for non-occluded region refinement.
        disp_corr_feat = self.encoder(disp,corr,disp_residual)
        non_occluded_regions_encoded_feature = torch.cat([cnn_input,disp_corr_feat],dim=1)
    
        # Using Global propagation here
        occluded_regions_encoded_feature = self.global_aggregator(attention,non_occluded_regions_encoded_feature)
        
        # Get the occlusion Mask and inverse occlusion mask
        inverse_occ_mask = torch.ones_like(occ).float() - occ
        inverse_occ_mask = inverse_occ_mask.detach()
        occ_mask = occ.detach()
        
        inp_final = occluded_regions_encoded_feature * occ_mask + non_occluded_regions_encoded_feature * inverse_occ_mask

        hidden_state = self.gru(hidden_state,inp_final)
        delta_disp = self.disp_head(hidden_state)
        mask = .25 * self.mask(hidden_state)
        mask = torch.clamp(mask,min=-1e3,max=1e3)
        
        return hidden_state,mask,delta_disp





class OcclusionAwarenessGlobalAggregationComplex14(nn.Module):
    def __init__(self,hidden_dim=128,
                 cost_volume_dimension=15,
                 prop_type='cspn'):
        super(OcclusionAwarenessGlobalAggregationComplex14,self).__init__()
        
        self.prop_type=prop_type
        # Cost Volune and depth aggregation
        self.encoder = PAMDepthEncoder(cost_volume_dimension=cost_volume_dimension)
        # GRU feature aggregation
        self.gru = SepConvGRU(hidden_dim=hidden_dim,input_dim=128+hidden_dim)
        # Global Aggregator
        self.global_aggregator = Aggregate(dim=256,heads=1,dim_head=256)
        
        # Local Propagation
        if prop_type=='cspn':
            self.local_aggregator = LocalAffinityPropagation(guidence_dimension=128,kernel_size=3,stride=1)

        
        
        # GRU feature aggreagtion
        self.gru = SepConvGRU(hidden_dim=hidden_dim,input_dim=128+hidden_dim)
        # Predict the disparity residual
        self.disp_head = DisparityHead(hidden_dim,hidden_dim=256)
        
        # Predict the upsample mask.
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16*9, 1, padding=0))
        
    
    def forward(self,hidden_state,
                cnn_input,
                corr,
                disp,
                occ,
                disp_residual,
                attention):
        
        # Encode the current disparity, current corr and current disparity residual for non-occluded region refinement.
        
        disp_corr_feat = self.encoder(disp,corr,disp_residual)
        non_occluded_regions_encoded_feature = torch.cat([cnn_input,disp_corr_feat],dim=1)
        
        
        # using the local propagation here
        non_occluded_regions_encoded_feature = self.local_aggregator(cnn_input,disp,non_occluded_regions_encoded_feature)
        
        # Using Global propagation here
        occluded_regions_encoded_feature = self.global_aggregator(attention,non_occluded_regions_encoded_feature)
        
        # using checkpoint for saving the memory
        #occluded_regions_encoded_feature = checkpoint(create_custom_self_attn(self.global_aggregator),attention,non_occluded_regions_encoded_feature)
        
        
        # Get the occlusion Mask and inverse occlusion mask
        inverse_occ_mask = torch.ones_like(occ).float() - occ
        inverse_occ_mask = inverse_occ_mask.detach()
        occ_mask = occ.detach()
        
        inp_final = occluded_regions_encoded_feature * occ_mask + non_occluded_regions_encoded_feature * inverse_occ_mask

        hidden_state = self.gru(hidden_state,inp_final)
        delta_disp = self.disp_head(hidden_state)
        
        mask = .25 * self.mask(hidden_state)
        mask = torch.clamp(mask,min=-1e3,max=1e3)
        
        return hidden_state,mask,delta_disp    








