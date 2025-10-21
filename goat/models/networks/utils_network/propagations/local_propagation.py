import torch
import torch.nn as nn
import torch.nn.functional as F
from goat.utils.core.disparity_warper import disp_warp, LRwarp_error

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

def SpatialPropagation(coarse_disp,offsets,offsets_pros,confidencemap=None,padding_mode='border'):
  grid = meshgrid(coarse_disp) #[B,2,H,W]
  sample_grid = grid + offsets #[B,2,H,W]
  sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
  moved_disp = F.grid_sample(coarse_disp, sample_grid, mode='bilinear', padding_mode=padding_mode)
  if confidencemap is not None:
    moved_confidenceMap = F.grid_sample(confidencemap,sample_grid,mode='bilinear',padding_mode=padding_mode)
  if confidencemap is None:
    refine_disp = moved_disp * offsets_pros + (torch.ones_like(offsets_pros)-offsets_pros)* coarse_disp
  else:
    refine_disp_sum = moved_disp * offsets_pros * confidencemap + (torch.ones_like(offsets_pros)-offsets_pros)* coarse_disp * moved_confidenceMap
    refine_disp = refine_disp_sum /(confidencemap+moved_confidenceMap)
  return refine_disp

def SpatialPropagationComplete(coarse_disp,offsets,offsets_pros, propagtion_nodes,confidencemap=None,padding_mode='border'):
    sum_disp = torch.zeros_like(coarse_disp)
    for i in range(propagtion_nodes):
        offset_single = offsets[:,i*2:(i+1)*2,:,:]
        offset_pros_single = offsets_pros[:,i,:,:].unsqueeze(1)
        refine_disp = SpatialPropagation(coarse_disp=coarse_disp,offsets=offset_single,offsets_pros=offset_pros_single,
        confidencemap=confidencemap,padding_mode=padding_mode)
        sum_disp+= refine_disp
    refine_disp = sum_disp/propagtion_nodes
    return refine_disp

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False, leaky_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             relu)
    return conv

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if stride!=1 or inplanes!= planes:
            if self.downsample is None:
                self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def compute_affinity_matrix(input_channel=9, feature_channel=32):
    affinity_matrix = nn.Sequential(
        nn.Conv2d(input_channel, 32, 3, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Conv2d(64, 8, 1, 1, 0, bias=False),
        nn.ReLU(True)
    )
    return affinity_matrix

def get_affinity_matrix(input_channels=9,feature_channel =32):
    affinity_matrix = BasicBlock(inplanes=input_channels,planes=feature_channel,stride=1,dilation=1)
    return affinity_matrix



class Affinity_propagation(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super(Affinity_propagation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.affinity_matrix_conv = compute_affinity_matrix(input_channel=12,feature_channel=32)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, normal_estimation,left_image,right_image,coarse_disparity):
        """
        params:
            normal_estimation: Estimated normal tensor with dim [N, 3, H, W] for the current version which may be later replaced 
                        by the concatenation of normal estimation and intermediate feature map.
            coarse_disparity: Estimated coarse disparity map with dim [N, 1, H, W].
        return:
            refined_disparity: Refined disparity after spatial propagation with the help of affinity matrix. The dim of
                         refined_disparity is [D, 1, H, W]
        """
        cur_left_img = F.interpolate(left_image,size=[coarse_disparity.size(-2),coarse_disparity.size(-1)], mode='bilinear',
                                        align_corners= False)
        cur_right_img = F.interpolate(right_image,size=[coarse_disparity.size(-2),coarse_disparity.size(-1)],mode='bilinear',
                                align_corners=False)
        disp_error =  LRwarp_error(cur_left_img,coarse_disparity,cur_right_img)

        input_features = torch.cat((normal_estimation,cur_left_img,cur_right_img,disp_error),dim=1)

        affinity_matrix = self.affinity_matrix_conv(input_features)
        gate = self._affinity_normalization(affinity_matrix)
        disparity_pad = self._disparity_pad(coarse_disparity)
        disparity_prop = torch.sum(gate * disparity_pad, dim=1, keepdim=True)
        disparity_prop = disparity_prop[:, :, 1:-1, 1:-1]
        disparity_prop = disparity_prop * 0.3 + coarse_disparity * 0.7
        assert disparity_prop.min() >= 0, "output shold be positive"
        return disparity_prop

    def _affinity_normalization(self, affinity_matrix):
        # transform the affinity matrix for spatial propagation
        # the affinity value of the 9-point neighboring region is preserved in the second dimension of affinity_matrix which is 8 for the current version
        gate1 = affinity_matrix.narrow(1, 0, 1)
        gate2 = affinity_matrix.narrow(1, 1*self.stride, self.stride)
        gate3 = affinity_matrix.narrow(1, 2*self.stride, self.stride)
        gate4 = affinity_matrix.narrow(1, 3*self.stride, self.stride)
        gate5 = affinity_matrix.narrow(1, 4*self.stride, self.stride)
        gate6 = affinity_matrix.narrow(1, 5*self.stride, self.stride)
        gate7 = affinity_matrix.narrow(1, 6*self.stride, self.stride)
        gate8 = affinity_matrix.narrow(1, 7*self.stride, self.stride)

        # pad affinity matrix with zero
        # top
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1 = left_top_pad(gate1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2 = center_top_pad(gate2)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3 = right_top_pad(gate3)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4 = left_center_pad(gate4)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5 = right_center_pad(gate5)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6 = left_bottom_pad(gate6)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7 = center_bottom_pad(gate7)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8 = right_bottom_pad(gate8)

        # N * 8 * H * W
        gate = torch.cat((gate1, gate2, gate3, gate4, gate5, gate6, gate7, gate8), dim=1)
        gate = self.softmax(gate)

        return gate
    
    def _disparity_pad(self, coarse_disparity):
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        disparity1 = left_top_pad(coarse_disparity)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        disparity2 = center_top_pad(coarse_disparity)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        disparity3 = right_top_pad(coarse_disparity)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        disparity4 = left_center_pad(coarse_disparity)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        disparity5 = right_center_pad(coarse_disparity)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        disparity6 = left_bottom_pad(coarse_disparity)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        disparity7 = center_bottom_pad(coarse_disparity)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        disparity8 = right_bottom_pad(coarse_disparity)
        
        disparity_pad = torch.cat((disparity1, disparity2, disparity3, disparity4, disparity5, disparity6, disparity7, disparity8), dim=1)
        return disparity_pad



# Local Affinity Propagation
class LocalAffinityPropagation(nn.Module):
    def __init__(self,guidence_dimension,kernel_size=3,stride=1,use_conf=False) -> None:
        super(LocalAffinityPropagation,self).__init__()
        self.use_conf = use_conf
        self.guidence_dimenson = guidence_dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.affinity_matrix_conv = compute_affinity_matrix(input_channel=self.guidence_dimenson+1,feature_channel=32)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,guidance_feature,disp,feature,conf=None):
        

        # Get the local Affinity matrix
        disp = disp.detach()
        inp_feature = torch.cat((guidance_feature,disp),dim=1)
        affinity_matrix = self.affinity_matrix_conv(inp_feature)
        
        # add confidence here
        if conf is not None:
            conf = conf.detach()
            affinity_matrix = affinity_matrix * conf        
        
        gate = self._affinity_normalization(affinity_matrix)
        gate = gate.unsqueeze(2)
        feature_pad = self._disparity_pad(feature)

        feature_prop = torch.sum(gate * feature_pad, dim=1)
        feature_prop = feature_prop[:, :, 1:-1, 1:-1]
        feature_prop + feature
        return feature_prop
        

    def _affinity_normalization(self, affinity_matrix):
        # transform the affinity matrix for spatial propagation
        # the affinity value of the 9-point neighboring region is preserved in the second dimension of affinity_matrix which is 8 for the current version
        gate1 = affinity_matrix.narrow(1, 0, 1)
        gate2 = affinity_matrix.narrow(1, 1*self.stride, self.stride)
        gate3 = affinity_matrix.narrow(1, 2*self.stride, self.stride)
        gate4 = affinity_matrix.narrow(1, 3*self.stride, self.stride)
        gate5 = affinity_matrix.narrow(1, 4*self.stride, self.stride)
        gate6 = affinity_matrix.narrow(1, 5*self.stride, self.stride)
        gate7 = affinity_matrix.narrow(1, 6*self.stride, self.stride)
        gate8 = affinity_matrix.narrow(1, 7*self.stride, self.stride)

        # pad affinity matrix with zero
        # top
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1 = left_top_pad(gate1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2 = center_top_pad(gate2)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3 = right_top_pad(gate3)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4 = left_center_pad(gate4)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5 = right_center_pad(gate5)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6 = left_bottom_pad(gate6)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7 = center_bottom_pad(gate7)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8 = right_bottom_pad(gate8)

        # N * 8 * H * W
        gate = torch.cat((gate1, gate2, gate3, gate4, gate5, gate6, gate7, gate8), dim=1)
        gate = self.softmax(gate)
#        gate_abs = torch.abs(gate)
#
#        abs_weight = torch.sum(gate_abs, dim=1, keepdim=True)+1e-9
#        gate = torch.div(gate, abs_weight)
#        gate_sum = torch.sum(gate, dim=1, keepdim=True)
        return gate
    
    def _disparity_pad(self, coarse_disparity):
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        disparity1 = left_top_pad(coarse_disparity)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        disparity2 = center_top_pad(coarse_disparity)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        disparity3 = right_top_pad(coarse_disparity)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        disparity4 = left_center_pad(coarse_disparity)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        disparity5 = right_center_pad(coarse_disparity)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        disparity6 = left_bottom_pad(coarse_disparity)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        disparity7 = center_bottom_pad(coarse_disparity)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        disparity8 = right_bottom_pad(coarse_disparity)
        
        disparity1 = disparity1.unsqueeze(1)
        disparity2 = disparity2.unsqueeze(1)
        disparity3 = disparity3.unsqueeze(1)
        disparity4 = disparity4.unsqueeze(1)
        disparity5 = disparity5.unsqueeze(1)
        disparity6 = disparity6.unsqueeze(1)
        disparity7 = disparity7.unsqueeze(1)
        disparity8 = disparity8.unsqueeze(1)
        
        disparity_pad = torch.cat((disparity1, disparity2, disparity3, disparity4, disparity5, disparity6, disparity7, disparity8), dim=1)
        return disparity_pad







# Local Propagation with confidence
class Affinity_propagation_with_conf(nn.Module):
    def __init__(self, input_dimension,kernel_size=3, stride=1):
        
        super(Affinity_propagation_with_conf, self).__init__()
        self.input_dimension = input_dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.affinity_matrix_conv = compute_affinity_matrix(input_channel=9+self.input_dimension,feature_channel=32)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, guidance_feature,left_image,right_image,coarse_disparity,confidence=None):
        """
        params:
            guidance_feature: Estimated normal tensor with dim [N, 3, H, W] for the 
            current version which may be later replaced 
            by the concatenation of normal estimation and intermediate feature map.
            coarse_disparity: Estimated coarse disparity map with dim [N, 1, H, W].
        return:
        refined_disparity: Refined disparity after spatial propagation with the help of affinity matrix. The dim of
                         refined_disparity is [D, 1, H, W]
        """
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
        
        # Disparity error + Left Image + right Image + guidance_feature
        input_features = torch.cat((guidance_feature,cur_left_img,cur_right_img,warped_error),dim=1)
        
        # Get the affinity Matrix
        affinity_matrix = self.affinity_matrix_conv(input_features)
        gate = self._affinity_normalization(affinity_matrix,confidence)
        disparity_pad = self._disparity_pad(coarse_disparity)
        disparity_prop = torch.sum(gate * disparity_pad, dim=1, keepdim=True)
        disparity_prop = disparity_prop[:, :, 1:-1, 1:-1]
        disparity_prop = disparity_prop * 0.3 + coarse_disparity * 0.7
        assert disparity_prop.min() >= 0, "output shold be positive"
        return disparity_prop

    # Affinity normalization
    def _affinity_normalization(self, affinity_matrix,confidence=None):
        # transform the affinity matrix for spatial propagation
        # the affinity value of the 9-point neighboring region is preserved in the second dimension of affinity_matrix which is 8 for the current version
        gate1 = affinity_matrix.narrow(1, 0, 1)
        gate2 = affinity_matrix.narrow(1, 1*self.stride, self.stride)
        gate3 = affinity_matrix.narrow(1, 2*self.stride, self.stride)
        gate4 = affinity_matrix.narrow(1, 3*self.stride, self.stride)
        gate5 = affinity_matrix.narrow(1, 4*self.stride, self.stride)
        gate6 = affinity_matrix.narrow(1, 5*self.stride, self.stride)
        gate7 = affinity_matrix.narrow(1, 6*self.stride, self.stride)
        gate8 = affinity_matrix.narrow(1, 7*self.stride, self.stride)

        # pad affinity matrix with zero
        # top
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1 = left_top_pad(gate1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2 = center_top_pad(gate2)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3 = right_top_pad(gate3)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4 = left_center_pad(gate4)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5 = right_center_pad(gate5)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6 = left_bottom_pad(gate6)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7 = center_bottom_pad(gate7)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8 = right_bottom_pad(gate8)
        # N * 8 * H * W: Affinity
        gate = torch.cat((gate1, gate2, gate3, gate4, gate5, gate6, gate7, gate8), dim=1)
        
        
        # Confidence 
        if confidence is not None:
            # Correlated with the Affinity 
            confidence1 = left_top_pad(confidence)
            confidence2 = center_top_pad(confidence)
            confidence3 = right_top_pad(confidence)
            confidence4 = left_center_pad(confidence)
            confidence5 = right_center_pad(confidence)
            confidence6 = left_bottom_pad(confidence)
            confidence7 = center_bottom_pad(confidence)
            confidence8 = right_bottom_pad(confidence)
            # Concated together
            confidence_pad = torch.cat((confidence1, confidence2, confidence3, confidence4, 
                                confidence5, confidence6, confidence7, confidence8), dim=1)
            # With confidence affinity
            gate = gate * confidence_pad

        # Here is the normalization
        gate = self.softmax(gate)
        return gate
 
    # def _confidence_pad(self,confidence,coarse_disparity):
    #     assert confidence!=None
    #     assert confidence.size(-2)==coarse_disparity.size(-2)
    #     # Top
    #     left_top_pad = nn.ZeroPad2d(0,2,0,2)
    #     confidence1 = left_top_pad(confidence)
    #     center_top_pad = nn.ZeroPad2d(1,1,0,2)
    #     confidence2 = center_top_pad(confidence)
    #     right_top_pad = nn.ZeroPad2d(2,0,0,2)
    #     confidence3 = right_top_pad(confidence)

    #     # Center
    #     left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
    #     confidence4 = left_center_pad(confidence)
    #     right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
    #     confidence5 = right_center_pad(confidence)

    #     # bottom
    #     left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
    #     confidence6 = left_bottom_pad(confidence)
    #     center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
    #     confidence7 = center_bottom_pad(confidence)
    #     right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
    #     confidence8 = right_bottom_pad(confidence)

    #     confidence_pad = torch.cat((confidence1, confidence2, confidence3, confidence4, 
    #                             confidence5, confidence6, confidence7, confidence8), dim=1)
    #     return confidence_pad

    def _disparity_pad(self, coarse_disparity):
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        disparity1 = left_top_pad(coarse_disparity)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        disparity2 = center_top_pad(coarse_disparity)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        disparity3 = right_top_pad(coarse_disparity)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        disparity4 = left_center_pad(coarse_disparity)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        disparity5 = right_center_pad(coarse_disparity)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        disparity6 = left_bottom_pad(coarse_disparity)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        disparity7 = center_bottom_pad(coarse_disparity)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        disparity8 = right_bottom_pad(coarse_disparity)
        
        disparity_pad = torch.cat((disparity1, disparity2, disparity3, disparity4, disparity5, disparity6, disparity7, disparity8), dim=1)
        return disparity_pad


# Feature Propagation
class FeatureAffinityPropagation(nn.Module):
    def __init__(self,kernel=3,stride=1,input_layer=12,feature_layer=32):
        super(FeatureAffinityPropagation,self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.feature_layer = feature_layer
        self.input_layer = input_layer
        self.affinity_matrix_conv = compute_affinity_matrix(input_channel=self.input_layer,feature_channel=32)
        self.kept= nn.Sequential(
            nn.Conv2d(self.feature_layer,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,1,1,1,0),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)
    
    def _affinity_normalization(self, affinity_matrix):
        gate1 = affinity_matrix.narrow(1, 0, 1)
        gate2 = affinity_matrix.narrow(1, 1*self.stride, self.stride)
        gate3 = affinity_matrix.narrow(1, 2*self.stride, self.stride)
        gate4 = affinity_matrix.narrow(1, 3*self.stride, self.stride)
        gate5 = affinity_matrix.narrow(1, 4*self.stride, self.stride)
        gate6 = affinity_matrix.narrow(1, 5*self.stride, self.stride)
        gate7 = affinity_matrix.narrow(1, 6*self.stride, self.stride)
        gate8 = affinity_matrix.narrow(1, 7*self.stride, self.stride)

        # pad affinity matrix with zero
        # top
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1 = left_top_pad(gate1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2 = center_top_pad(gate2)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3 = right_top_pad(gate3)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4 = left_center_pad(gate4)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5 = right_center_pad(gate5)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6 = left_bottom_pad(gate6)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7 = center_bottom_pad(gate7)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8 = right_bottom_pad(gate8)

        # N * 8 * H * W
        gate = torch.cat((gate1, gate2, gate3, gate4, gate5, gate6, gate7, gate8), dim=1)
        gate = self.softmax(gate)
        return gate

    def _feature_pad(self, coarse_disparity):
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        disparity1 = left_top_pad(coarse_disparity)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        disparity2 = center_top_pad(coarse_disparity)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        disparity3 = right_top_pad(coarse_disparity)

        # center
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        disparity4 = left_center_pad(coarse_disparity)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        disparity5 = right_center_pad(coarse_disparity)

        # bottom
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        disparity6 = left_bottom_pad(coarse_disparity)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        disparity7 = center_bottom_pad(coarse_disparity)
        right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        disparity8 = right_bottom_pad(coarse_disparity)
        
        feature_pad = torch.cat((disparity1, disparity2, disparity3, disparity4, disparity5, disparity6, disparity7, disparity8), dim=1)
        return feature_pad

    def forward(self,normal_estimation,left_image,right_image,disp_error,features):
        kept_rate = self.kept(features)
        input_features = torch.cat((normal_estimation,left_image,right_image,disp_error),dim=1)
        affinity_matrix = self.affinity_matrix_conv(input_features)
        gate = self._affinity_normalization(affinity_matrix)
        feature_pad = self._feature_pad(features)
        cg = gate.shape[1]
        cf = feature_pad.shape[1]
        gate = gate.view(gate.shape[0],8,cg//8,gate.shape[-2],gate.shape[-1])
        feature_pad = feature_pad.view(feature_pad.shape[0],8,cf//8,feature_pad.shape[-2],feature_pad.shape[-1])
        feature_prop = torch.sum(gate *feature_pad,dim=1)
        feature_prop = feature_prop[:, :, 1:-1, 1:-1]
        feature_prop = feature_prop * (1.0-kept_rate) + features * kept_rate
        return feature_prop
        

if __name__=="__main__":
    left_image = torch.randn(2,3,100,100)
    right_image = torch.randn(2,3,100,100)
    corase_disp = torch.abs(torch.randn(2,1,100,100))

    confidenceMap = torch.randn(2,1,100,100)
    guidance_feature = torch.randn(2,128,100,100)
    affinity_propagation = Affinity_propagation_with_conf(kernel_size=3,stride=1,input_dimension=128)
    # refine disparity
    refined_disparity = affinity_propagation(guidance_feature,left_image,right_image,corase_disp,None)
    print(refined_disparity.shape)