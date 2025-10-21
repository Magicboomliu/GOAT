"""
Dataset loaders for various stereo matching datasets
"""

# KITTI loaders
from .kitti.kitti_loader import *
from .kitti.kitti_loader_occ import *
from .kitti.SceneflowLoader import *
from .kitti.SceneflowLoaderOcc import *
from .kitti.middleburry_loader import *

# FAT loaders
from .fat.FAT_dataset import *

