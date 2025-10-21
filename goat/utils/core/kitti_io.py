from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys

# load psedo GT
def load_psedo_kitti(path):
    return np.load(path)


def default_loader(path):
    return Image.open(path).convert('RGB')

def converter(image_type):
    img = np.array(image_type).astype(np.float32)
    return img


# Read Image
def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img

# Read KITTI Image by Two Step
def read_kitti_image_step1(filename):
    img = Image.open(filename).convert('RGB')
    return img

def read_kitti_image_step2(img):
    img = np.array(img).astype(np.float32)
    return img


# Read Slant Window
def read_slant(filename):
    # read npy from file
    slant_window = np.load(filename)
    
    return slant_window
    

# Read Disp
def read_disp(filename, subset=False):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
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


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)


def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    #depth = np.ascontiguousarray(depth,dtype=np.float32)/256.
    depth = depth.astype(np.float32) / 256.
    return depth

def read_kitti_step1(filename):
    return Image.open(filename)

def read_kitti_step2(data):
    depth = np.array(data)
    depth = depth.astype(np.float32) / 256.
    return depth



def read_occ(data):
    occ = np.load(data)
    
    return occ