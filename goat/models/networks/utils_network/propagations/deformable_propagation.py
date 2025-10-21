import torch
import torch.nn.functional as F
import torch.nn as nn
from third_party.deform.modules.modulated_deform_conv import ModulatedDeformConvPack, ModulatedDeformConvFunction
from goat.utils.core.disparity_warper import disp_warp


'''Args Settings'''
class Args(object):
    def __init__(self,prop_times=1,affinity='TGASS',nlk=3,affinity_gamma=1.0,
                    conf_prop=True,
                    fusion =True) -> None:
        # propagation times.
        self.prop_times = prop_times
        # affinity type.
        self.affinity = affinity
        self.nlk = nlk
        # gamma.
        self.affinity_gamma = affinity_gamma
        # using confidence
        self.conf_prop = conf_prop
        # dispairty refusion
        self.fusion = fusion
        
'''Compute the Affinity'''
def compute_affinity_matrix(input_channel=9, neighbour_num=8,non_local_kernel=3):
    affinity_matrix = nn.Sequential(
        nn.Conv2d(input_channel, 32, non_local_kernel, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Conv2d(32, 64, non_local_kernel, 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Conv2d(64, 3*neighbour_num, non_local_kernel, 1, 1, bias=False),
    )
    return affinity_matrix
        
'''Non-Local Spatial Propagation'''
class NonLocalSpatialPropagation(nn.Module):
    def __init__(self,args:Args,guidance_channel):
        super(NonLocalSpatialPropagation,self).__init__()
        # Configs
        self.args = args
        self.prop_times = self.args.prop_times
        self.affinity = self.args.affinity
        self.non_local_kernel = self.args.nlk
        # neighbours numbers
        self.num = self.non_local_kernel * self.non_local_kernel -1
        self.idx_ref = self.num // 2
        # input channels
        self.guidance_channel = guidance_channel

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            # get local affinities
            self.conv_offset_aff = compute_affinity_matrix(input_channel=self.guidance_channel,
            neighbour_num=self.num,non_local_kernel=self.non_local_kernel)

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((1, 1, 3, 3)))
        self.b = nn.Parameter(torch.zeros(1))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.deformable_groups = 1
        self.im2col_step = 64
    
    # Guidance ---> non-local affinity
    def _get_offset_affinity(self, guidance, confidence=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)
            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.non_local_kernel
                hh = idx_off // self.non_local_kernel

                if ww == (self.non_local_kernel - 1) / 2 and hh == (self.non_local_kernel - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()
                # Confidence propagation
                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            # confidence * affinity
            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat
    
    def forward(self,coarse_disparity,normal_estimation,left_image,right_image,confidence=None):
        # Current Scale of the disparity
        current_scale = left_image.size(-2)//coarse_disparity.size(-2)
        
        # resize the image
        cur_left_img = F.interpolate(left_image,size=[coarse_disparity.size(-2),
                                                 coarse_disparity.size(-1)], mode='bilinear',
                                        align_corners= False)
        cur_right_img = F.interpolate(right_image,size=[coarse_disparity.size(-2),
                                       coarse_disparity.size(-1)],mode='bilinear',
                                align_corners=False)
        disp_ = coarse_disparity / current_scale
        warped_right = disp_warp(cur_right_img,disp_)[0]
        warped_error = warped_right - cur_left_img
        # Disparity error + Left Image + right Image + surface normal = 12
        guidance = torch.cat((normal_estimation,cur_left_img,cur_right_img,warped_error),dim=1)
        if self.args.conf_prop:
            assert confidence is not None
        
        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance,confidence)
        else:
            offset,aff = self._get_offset_affinity(guidance,None)
        
        # Weight
        center_aff = aff.shape[1]
        center_aff_idx = center_aff//2
        center_affinites = aff[:,center_aff_idx,:,:]
        others_affinites = torch.ones_like(center_affinites).type_as(center_affinites) - center_affinites
        others_affinites = torch.clamp(others_affinites,max=1.0,min=0.0)
        
        
        inter_disp= coarse_disparity
        final_disp = coarse_disparity
        # Propagation
        for k in range(1,self.prop_times+1):
            inter_disp = self._propagate_once(inter_disp,offset,aff)
            mask = inter_disp>=0
            inter_disp = inter_disp * mask
        
        inter_disp = torch.clamp(inter_disp,min=0)
        if self.args.fusion:
            final_disp = center_affinites.detach() * coarse_disparity + others_affinites.detach() * inter_disp
        else:
            final_disp = inter_disp

        final_disp = torch.clamp(final_disp,min=0) 
        
        return final_disp


class Args(object):
    def __init__(self,prop_times=1,affinity='TGASS',nlk=3,affinity_gamma=1.0,
                    conf_prop=True,
                    fusion =True) -> None:
        self.prop_times = prop_times
        self.affinity = affinity
        self.nlk = nlk
        self.affinity_gamma = affinity_gamma
        self.conf_prop = conf_prop
        self.fusion = fusion
        
'''Non-Local Feature Propagation'''
class NonLocalFeaturePropagation(nn.Module):
    def __init__(self,args:Args,guidance_channel,feature_channel=1):
        super(NonLocalFeaturePropagation,self).__init__()
        
        self.args = args
        self.feature_channels = feature_channel
        self.prop_times = self.args.prop_times
        self.affinity = self.args.affinity
        self.non_local_kernel = self.args.nlk
        # neighbours numbers
        self.num = self.non_local_kernel * self.non_local_kernel -1
        self.idx_ref = self.num // 2
        # input channels
        self.guidance_channel = guidance_channel

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            # get local affinities
            self.conv_offset_aff = compute_affinity_matrix(input_channel=self.guidance_channel,
            neighbour_num=self.num,non_local_kernel=self.non_local_kernel)

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((1, feature_channel, 3, 3)))
        self.b = nn.Parameter(torch.zeros(1))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.deformable_groups = 1
        self.im2col_step = 64
    
    # Guidance ---> non-local affinity
    def _get_offset_affinity(self, guidance, confidence=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)
            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()
            
            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.non_local_kernel
                hh = idx_off // self.non_local_kernel
                if ww == (self.non_local_kernel - 1) / 2 and hh == (self.non_local_kernel - 1) / 2:
                    continue
                offset_tmp = offset_each[idx_off].detach()
                # Confidence propagation
                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            # confidence * affinity
            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    # Non-Local Propagation
    def _propagate_once(self, feat, offset, aff):

        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )
        return feat
    
    def forward(self,disp,guidance_feature,feature,confidence=None):
        
        # Guidance feature here
        guidance = torch.cat((guidance_feature,disp),dim=1)
        

        # whether using the confidence
        if self.args.conf_prop:
            assert confidence is not None
        
        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance,confidence)
        else:
            offset,aff = self._get_offset_affinity(guidance,None)
        
        
        
        # Weight
        center_aff = aff.shape[1]
        center_aff_idx = center_aff//2
        center_affinites = aff[:,center_aff_idx,:,:]
        others_affinites = torch.ones_like(center_affinites).type_as(center_affinites) - center_affinites
        others_affinites = torch.clamp(others_affinites,max=1.0,min=0.0)
        
        
        # Initial Feature
        inter_feature = feature
        final_feature = feature
        # Feature Propagation Here
        for k in range(1,self.prop_times+1):
            # update the inter featuure
            inter_feature= self._propagate_once(inter_feature,offset,aff)
                  
        if self.args.fusion:
            final_feature = center_affinites.detach() * inter_feature + others_affinites.detach() * final_feature
        else:
            final_feature = final_feature

        return final_feature


if __name__=="__main__":
    prop_args = Args(prop_times=1,affinity='TGASS',nlk=3,affinity_gamma=0.5,conf_prop=False,fusion=True)
    nlsp = NonLocalFeaturePropagation(prop_args,guidance_channel=129,feature_channel=2).cuda()
    
    disp = torch.randn(1,1,40,80).cuda()
    guidance_feature = torch.randn(1,128,40,80).cuda()
    feat = torch.randn(1,2,40,80).cuda()
    
    final_feature = nlsp(disp,guidance_feature,feat)
    
    pass





