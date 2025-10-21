import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import skimage


# Get Normal from Zhang Songyan
def get_normal(target_disp):
    edge_kernel_x = torch.from_numpy(np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])).type_as(target_disp)
    edge_kernel_y = torch.from_numpy(np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])).type_as(target_disp)
    sobel_kernel = torch.cat((edge_kernel_x.view(1, 1, 3, 3), edge_kernel_y.view(1, 1, 3, 3)), dim=0)
    sobel_kernel.requires_grad = False
    grad_depth = torch.nn.functional.conv2d(target_disp, sobel_kernel, padding=1) * -1.
    N, C, H, W = grad_depth.shape
    norm = torch.cat((grad_depth, torch.ones(N, 1, H, W).to(target_disp.device)), dim=1)
    target_normal = F.normalize(norm, dim=1)
    return target_normal

# Read disp and Normal
def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def main():
    imgL_ori = (skimage.io.imread("left.png"))
    disp_gt,scales = read_pfm('disp.pfm')

    dispL = torch.from_numpy(disp_gt.copy()).float()
    dispL = dispL.unsqueeze(0)
    dispL = dispL.unsqueeze(1)
    target_normal = get_normal(dispL)
    vis_normal = target_normal.squeeze(0)
    target_normal = vis_normal.permute(1,2,0).numpy()
    print(target_normal.shape)
    # plt.imshow(target_normal)
    # plt.show()


if __name__=="__main__":
    main()