import torch
import torch.nn as nn
import torch.nn.functional as F

class CostVolume(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures
        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()

        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()

        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature

        elif self.feature_similarity == 'concat':
            cost_volume = left_feature.new_zeros(b, 2 * c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                            dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)

        elif self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                                right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        elif self.feature_similarity == 'correlation_wo_mean':
            cost_volume = left_feature.new_zeros(b, c,self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :,i, :, i:] = (left_feature[:, :, :, i:] *
                                                right_feature[:, :, :, :-i])
                else:
                    cost_volume[:, :,i, :, :] = (left_feature * right_feature)
        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume
