import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self,
                 input_dim = 128):
        super(SpatialAttention,self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(input_dim+3, input_dim//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_dim//2),
            nn.ReLU(True),
            nn.Conv2d(input_dim//2, input_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_dim//2),
            nn.ReLU(True),
            nn.Conv2d(input_dim//2, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,warped_feature_map=None,warped_rgb_map=None):
        
        error_feature  = torch.cat((warped_feature_map,warped_rgb_map),dim=1)
        attention = self.attention(error_feature)
        return attention


class SpatialAttentionV1(nn.Module):
    def __init__(self,
                 input_dim =3,
                 hidden_dim =32):
        super(SpatialAttentionV1,self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,warped_rgb_map=None):
        attention = self.attention(warped_rgb_map)
        return attention