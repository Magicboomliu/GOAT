import torch
import torch.nn as nn
import torch.nn.functional as F


# Parallax-Attention Block
class PABV2(nn.Module):
    def __init__(self, channels,scale=1):
        super(PABV2, self).__init__()
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),

        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),

        )

    def forward(self, x_left, x_right, cost):
        '''
        :param x_left:      features from the left image  (B * C * H * W)
        :param x_right:     features from the right image (B * C * H * W)
        :param cost:        input matching cost           (B * H * W * W)
        '''
        b, c, h, w = x_left.shape
        fea_left = x_left
        fea_right = x_right
        
        clamped_range = int(w//3)

        Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()                     # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3) .contiguous()                     # B * H * C * W
        
        cost_right2left = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_right2left = cost_right2left - torch.tril(cost_right2left, -clamped_range)
        cost_right2left = torch.tril(cost_right2left) # Downer Traniagle        
        cost_right2left = (cost_right2left + cost[0])

        # C_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1).contiguous()                    # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3).contiguous()                       # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_left2right = cost_left2right - torch.triu(cost_left2right, clamped_range)
        cost_left2right  = torch.triu(cost_left2right) # Upper traniagle
        cost_left2right = (cost_left2right + cost[1])

        return x_left + fea_left, \
               x_right + fea_right, \
               (cost_right2left, cost_left2right)
               

class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right, cost):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)

        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


class PAM_stageV2(nn.Module):
    def __init__(self, channels):
        super(PAM_stageV2, self).__init__()
        self.pab1 = PABV2(channels)
        self.pab2 = PABV2(channels)
        self.pab3 = PABV2(channels)
        self.pab4 = PABV2(channels)

    def forward(self, fea_left, fea_right, cost):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)

        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)


        return fea_left, fea_right, cost




