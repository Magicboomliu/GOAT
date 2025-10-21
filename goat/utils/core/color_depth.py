import cv2
from PIL import Image
import os
import argparse



def depth2color(depth_img, save_dir):
    depth_img = cv2.imread(depth_img)
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=2), cv2.COLORMAP_JET)    # JET RAINBOW
    # depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)    # JET RAINBOW
    # depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    depth_color = Image.fromarray(depth_color)
    depth_color.save(save_dir)



if __name__=='__main__':
    color_dir ="/home/zliu/Cur_Networks/OGMNet/analysis_results/KITTI2015_SubmissionCCC"
    depth_dir = "/home/zliu/Cur_Networks/OGMNet/analysis_results/KITTI2015_Submission"
    if not os.path.exists(color_dir):
        os.mkdir(color_dir)
    for img in os.listdir(depth_dir):
        print('writing image ', img)
        depth_img = os.path.join(depth_dir, img)
        # print(depth_img)
        depth2color(depth_img, os.path.join(color_dir, img))