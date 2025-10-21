import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from utils import read_text_lines
from file_io import read_disp,read_img
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def read_img_2014(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB'))
    return img

def remove_NAN_INF(tensor):
    tensor = torch.where(torch.isnan(tensor),torch.zeros_like(tensor).type_as(tensor),tensor)
    tensor = torch.where(torch.isinf(tensor),torch.zeros_like(tensor).type_as(tensor),tensor)
    
    return tensor


if __name__=="__main__":
    
    mid_2014_list = "/media/zliu/datagrid1/liu/MIDDD/datasets/Middlebury/2014"
    
    # folder = os.listdir(mid_2014_list)
    
    
    # # Middleburry 2014 List
    # perfect_folder_names = []
    # for f in folder:
    #     if "imperfect" in f:
    #         perfect_folder_names.append(f)
    #     else:
    #         #perfect_folder_names.append(f)
    #         continue
    
    # saved_lines =[]
    # # left/right image, left_disp,right_disp
    # for p in perfect_folder_names:
    #     left_image = os.path.join(p,"im0.png")
    #     right_img = os.path.join(p,"im1.png")
    #     right_img_E = os.path.join(p,"im1E.png")
    #     right_img_L = os.path.join(p,"img1L.png")
    #     disp_left = os.path.join(p,'disp0.pfm')
    #     disp_right = os.path.join(p,'disp1.pfm')
        
    #     saved_line = left_image +" "+ right_img + " "+ right_img_E +" "+ right_img_L +" "+ disp_left +" " +disp_right
    #     saved_lines.append(saved_line)
    
    # with open("mid_2014_im.list",'w') as f:
    #     for id,line in enumerate(saved_lines):
    #         if id !=len(saved_lines)-1:
    #             f.writelines(line+"\n")
    #         else:
    #             f.writelines(line)
    
    
    instance_left = "/media/zliu/datagrid1/liu/MIDDD/datasets/Middlebury/MiddEval3/trainingH/Adirondack/im0.png"
    instance_right = "/media/zliu/datagrid1/liu/MIDDD/datasets/Middlebury/MiddEval3/trainingH/Adirondack/im1.png"
    instance_occ = "/media/zliu/datagrid1/liu/MIDDD/datasets/Middlebury/MiddEval3/trainingH/Adirondack/disp0GT.pfm"
    
    middleburry_list = "mid_2014_im.list"
    
    lines = read_text_lines(middleburry_list)
    
    for idx, line in enumerate(lines):
        print(line)
        line = line.split()
        left_img_path = os.path.join(mid_2014_list,line[0])
        right_img_path = os.path.join(mid_2014_list,line[1])
        right_img_path_E = os.path.join(mid_2014_list,line[2])
        right_img_path_L = os.path.join(mid_2014_list,line[3])
        left_disp = os.path.join(mid_2014_list,line[4])
        right_disp = os.path.join(mid_2014_list,line[5])
        
        left_img = read_img_2014(left_img_path)
        right_img = read_img_2014(right_img_path)
        
        left = np.transpose(left_img,(2, 0, 1)).astype(np.float32)  # [3, H, W]
        left = torch.from_numpy(left)
        left = left.unsqueeze(0)
        left = F.interpolate(left,scale_factor=1/2,mode='bilinear',align_corners=False)
        left_resize = left.squeeze(0).permute(1,2,0).cpu().numpy()
        
        disp_gt = read_disp(left_disp)
        disp_gt = torch.from_numpy(disp_gt)
        disp_gt = disp_gt.unsqueeze(0).unsqueeze(0)
        # print(disp_gt.shape)
        disp_gt = F.interpolate(disp_gt,scale_factor=1./2,mode='nearest') *0.5
        
        disp_gt = remove_NAN_INF(disp_gt)
        disp_gt = disp_gt.squeeze(0).squeeze(0).cpu().numpy()
        
        
        
        # left_img_pil = Image.fromarray(left_img)
        # H,W = left_img_pil.size
        # left_img_pil_resize = left_img_pil.resize([H//2,W//2])
        # left_img_resize = np.array(left_img_pil_resize).astype(np.float32)
        
        left_img_eval = read_img(instance_left)
        disp_gt_eval = read_disp(instance_occ)
        disp_gt_eval_tensor = torch.from_numpy(disp_gt_eval).unsqueeze(0).unsqueeze(0)
        disp_gt_eval_tensor = remove_NAN_INF(disp_gt_eval_tensor)
        disp_gt_eval = disp_gt_eval_tensor.squeeze(0).squeeze(0).cpu().numpy()
        

        
        print(np.abs(left_resize-left_img_eval).mean())
        

        print(np.abs(disp_gt_eval-disp_gt).mean())
        
        # print(left_img_resize[20:22,20:22,:])
        # print("-------------------------------")
        # print(left_img_eval[20:22,20:22,:])
        
        break
