#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-04-18 17:05:57

import cv2
import scipy.io as sio
from pathlib import Path
import commentjson as json
from skimage import img_as_float32, img_as_ubyte
from train import Trainer
from utils_bsrdm import update_args, degradation, set_seed, imshow, center_crop

import argparse
parser = argparse.ArgumentParser(description='Parameter configurations')
parser.add_argument('--gpu_id', type=int, default=2, help='GPU Index, (default: 2)')
parser.add_argument('--rho', type=float, default=0.2, help='Hyper-parameter rho, (default: 0.2)')

# degradation settings
parser.add_argument('--sf', type=int, default=2, help='Scale factor: 2, 3, or 4, (default: 2)')
parser.add_argument('--noise_type', type=str, default='Gaussian',
                                         help='Noise type: Gaussian or Signal, (default: Gaussian)')
parser.add_argument('--noise_level', type=float, default=2.55,
                                             help='Noise Level for Gaussian Noise, (default: 2.55)')
parser.add_argument('--noise_estimator', type=str, default='iid',
                                        help='Noise estimation method: niid or iid, (default: iid)')
parser.add_argument('--downsampler', type=str, default='direct',
                                   help='Downsampler setting: bicubic or direct, (default: direct)')
args = parser.parse_args()

def main():
    # load the default options
    with open('./options/options1.json', 'r') as f:
        opts_json = json.load(f)
    # update part of the default options according to your give setting
    update_args(opts_json, args)

    # load HR image
    im_path_HR = './testsets/Set14/lenna.bmp'
    im_HR = cv2.imread(str(im_path_HR), flags=cv2.IMREAD_UNCHANGED)   # [0, 255], uint8
    im_HR = img_as_float32(im_HR)                                     # [0, 1.0], float
    if im_HR.ndim == 3:
        im_HR = im_HR[:, :, ::-1]

    # load the blur kernel
    kernel_path = Path('./testsets/kernels_synthetic') / ('kernels_x'+str(args.sf) + '.mat')
    kernel = sio.loadmat(str(kernel_path))['kernels'][:, :, 3]

    set_seed(args.seed)
    im_LR = degradation(im_HR, args.sf, kernel,
                        noise_level = args.noise_level,
                        convolve = False,
                        noise_type = args.noise_type,
                        downsampler = args.downsampler)
    trainer_sisr = Trainer(args, im_LR, im_HR, kernel)
    trainer_sisr.train()
    im_sr = trainer_sisr.get_HR_res()
    kernel_est = trainer_sisr.get_kernel_est()
    imshow(im_LR, title='LR Image')
    imshow(im_sr, title='SR Image')

if __name__ == '__main__':
    main()

