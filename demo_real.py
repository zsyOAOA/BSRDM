#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-04-18 17:05:57

import cv2
from pathlib import Path
from train import Trainer
import commentjson as json
from skimage import img_as_float32
from utils_bsrdm import update_args

import argparse
parser = argparse.ArgumentParser(description='Parameter configurations')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU Index, (default: 0)')
parser.add_argument('--sf', type=int, default=2, help='Scale factor for SISR, (default: 2)')
parser.add_argument('--disp', type=int, default=0,
                                        help='Whether displaying the log information, (default: 0)')

args = parser.parse_args()

def main():
    # load the default settings
    with open('./options/options1.json', 'r') as f:
        opts_json = json.load(f)
    # update the default settings according to your input
    update_args(opts_json, args)

    # folder to save the results
    save_dir = Path('./testsets') / ('RealSRSet_BSRDM_x'+str(args.sf))
    if not save_dir.exists():
        save_dir.mkdir()

    im_path_list = sorted([x for x in Path('./testsets/RealSRSet').glob('*.png')])
    for ii, im_path in enumerate(im_path_list):
        im_name = im_path.stem
        if im_name in ['frog', 'painting', 'ppt3', 'tiger']:
            args.rho = 0.1
        elif im_name in ['oldphoto3']:
            args.rho = 0.4
        else:
            args.rho = 0.2

        print('{:02d}/{:02d}: Image: {:15s}, sf: {:d}, rho={:3.1f}'.format(ii+1, len(im_path_list), im_name, args.sf, args.rho))

        im_LR = cv2.imread(str(im_path), flags=cv2.IMREAD_UNCHANGED)   # [0, 255], uint8
        im_LR = img_as_float32(im_LR)                                  # [0, 1.0], float32
        if im_LR.ndim == 3:
            im_LR = im_LR[:, :, ::-1].copy()
        trainer_sisr = Trainer(args, im_LR)
        trainer_sisr.train()
        im_HR_est = trainer_sisr.get_HR_res()
        save_path = save_dir / (im_name + '_BSRDM.png')
        cv2.imwrite(str(save_path), im_HR_est[:, :, ::-1])

if __name__ == '__main__':
    main()

