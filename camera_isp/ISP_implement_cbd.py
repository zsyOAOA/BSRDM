# -*- coding: utf-8 -*-
# Power by Zongsheng Yue 2019-06-11 22:29:36
'''
ISP class
=============================================================================================================
include main operators of in-camera processing pipeline (ISP) and some corresponding inverse operators.

Reference: https://github.com/GuoShi28/CBDNet

'''
import cv2
import torch
import random
import skimage
import numpy as np
import os.path as oph
import scipy.io as sio
from math import log, exp
import pyximport; pyximport.install()
from .noise_synthetic.tone_mapping_cython import CRF_Map_Cython, ICRF_Map_Cython
from .noise_synthetic.Demosaicing_malvar2004 import demosaicing_CFA_Bayer_Malvar2004

class ISP:
    def __init__(self, curve_path=None):
        if curve_path is None:
            curve_path = oph.join(oph.dirname(oph.abspath(__file__)), 'noise_synthetic')
        filename = oph.join(curve_path, '201_CRF_data.mat')
        CRFs = sio.loadmat(filename)
        self.I = CRFs['I']
        self.B = CRFs['B']
        filename = oph.join(curve_path, 'dorfCurvesInv.mat')
        inverseCRFs = sio.loadmat(filename)
        self.I_inv = inverseCRFs['invI']
        self.B_inv = inverseCRFs['invB']
        self.xyz2cam_all = np.array([[1.0234,-0.2969,-0.2266,-0.5625,1.6328,-0.0469,-0.0703,0.2188,0.6406],
                                     [0.4913,-0.0541,-0.0202,-0.613,1.3513,0.2906,-0.1564,0.2151,0.7183],
                                     [0.838,-0.263,-0.0639,-0.2887,1.0725,0.2496,-0.0627,0.1427,0.5438],
                                     [0.6596,-0.2079,-0.0562,-0.4782,1.3016,0.1933,-0.097,0.1581,0.5181]]
                                    )

    def ICRF_Map(self, img, index=0):
        invI_temp = self.I_inv[index, :]
        invB_temp = self.B_inv[index, :]
        out = ICRF_Map_Cython(img.astype(np.float32), invI_temp.astype(np.float32), invB_temp.astype(np.float32))
        return out

    def CRF_Map(self, img, index=0):
        I_temp = self.I[index, :]  # shape: (1024, 1)
        B_temp = self.B[index, :]  # shape: (1024, 1)
        out = CRF_Map_Cython(img.astype(np.float32), I_temp.astype(np.float32), B_temp.astype(np.float32))
        return out

    def RGB2XYZ(self, img):
        xyz = skimage.color.rgb2xyz(img)
        return xyz

    def XYZ2RGB(self, img):
        rgb = skimage.color.xyz2rgb(img)
        return rgb

    def XYZ2CAM(self, img, M_xyz2cam=0):
        if type(M_xyz2cam) is int:
            # cam_index = np.random.random((1, 4))
            cam_index = torch.rand((1,4)).numpy()
            cam_index = cam_index / np.sum(cam_index)
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0, 0] + \
                         self.xyz2cam_all[1, :] * cam_index[0, 1] + \
                         self.xyz2cam_all[2, :] * cam_index[0, 2] + \
                         self.xyz2cam_all[3, :] * cam_index[0, 3] \
                         )
            self.M_xyz2cam = M_xyz2cam

        M_xyz2cam = np.reshape(M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        cam = self.apply_cmatrix(img, M_xyz2cam)
        cam = np.clip(cam, 0, 1)
        return cam

    def CAM2XYZ(self, img, M_xyz2cam=0):
        if type(M_xyz2cam) is int:
            cam_index = torch.rand((1,4)).numpy()
            cam_index = np.array([random.random(), random.random(), random.random(), random.random()]).reshape((1,4))
            cam_index = cam_index / np.sum(cam_index)
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0, 0] +
                         self.xyz2cam_all[1, :] * cam_index[0, 1] +
                         self.xyz2cam_all[2, :] * cam_index[0, 2] +
                         self.xyz2cam_all[3, :] * cam_index[0, 3]
                         )
        M_xyz2cam = np.reshape(M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        M_cam2xyz = np.linalg.inv(M_xyz2cam)
        xyz = self.apply_cmatrix(img, M_cam2xyz)
        xyz = np.clip(xyz, 0, 1)
        return xyz

    def apply_cmatrix(self, img, matrix):
        r = (matrix[0, 0] * img[:, :, 0] + matrix[0, 1] * img[:, :, 1]
             + matrix[0, 2] * img[:, :, 2])
        g = (matrix[1, 0] * img[:, :, 0] + matrix[1, 1] * img[:, :, 1]
             + matrix[1, 2] * img[:, :, 2])
        b = (matrix[2, 0] * img[:, :, 0] + matrix[2, 1] * img[:, :, 1]
             + matrix[2, 2] * img[:, :, 2])
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        results = np.concatenate((r, g, b), axis=2)
        return results

    def BGR2RGB(self, img):
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        return rgb_img

    def RGB2BGR(self, img):
        r, g, b = cv2.split(img)
        bgr_img = cv2.merge([b, g, r])
        return bgr_img

    def mosaic_bayer(self, rgb, pattern='BGGR'):
        # analysis pattern
        num = np.zeros(4, dtype=int)
        # the image store in OpenCV using BGR
        temp = list(self.find(pattern, 'R'))
        num[temp] = 0
        temp = list(self.find(pattern, 'G'))
        num[temp] = 1
        temp = list(self.find(pattern, 'B'))
        num[temp] = 2

        mosaic_img = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=rgb.dtype)
        mosaic_img[0::2, 0::2] = rgb[0::2, 0::2, num[0]]
        mosaic_img[0::2, 1::2] = rgb[0::2, 1::2, num[1]]
        mosaic_img[1::2, 0::2] = rgb[1::2, 0::2, num[2]]
        mosaic_img[1::2, 1::2] = rgb[1::2, 1::2, num[3]]
        return mosaic_img

    def WB_Mask(self, img, pattern, fr_now, fb_now):
        wb_mask = np.ones(img.shape)
        if  pattern == 'RGGB':
            wb_mask[0::2, 0::2] = fr_now
            wb_mask[1::2, 1::2] = fb_now
        elif  pattern == 'BGGR':
            wb_mask[1::2, 1::2] = fr_now
            wb_mask[0::2, 0::2] = fb_now
        elif  pattern == 'GRBG':
            wb_mask[0::2, 1::2] = fr_now
            wb_mask[1::2, 0::2] = fb_now
        elif  pattern == 'GBRG':
            wb_mask[1::2, 0::2] = fr_now
            wb_mask[0::2, 1::2] = fb_now
        return wb_mask

    def find(self, str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    def Demosaic(self, bayer, pattern='BGGR'):
        results = demosaicing_CFA_Bayer_Malvar2004(bayer, pattern)
        results = np.clip(results, 0, 1)
        return results

    def add_PG_noise(self, bayer_img):
        '''
        reference: Unprocessing Images for Learned Rar Denoising

        '''
        log_min_shot_level = log(0.0001)
        log_max_shot_level = log(0.012)
        log_shot_level = log_min_shot_level + (1/4)*(log_max_shot_level - log_min_shot_level)
        shot_level = exp(log_shot_level)

        line = lambda x: 2.18 * x + 1.20
        log_read_level = line(log_shot_level) + random.gauss(mu=0, sigma=0.26)
        read_level = exp(log_read_level)

        variance = bayer_img * shot_level + read_level
        noise = torch.randn(bayer_img.shape).numpy() * np.sqrt(variance)
        bayer_img_noisy = bayer_img + noise
        return bayer_img_noisy

    def noise_generate_srgb(self, img):
        img_rgb = img
        # -------- INVERSE ISP PROCESS -------------------
        # Step 1 : inverse tone mapping
        icrf_index = random.randint(0, 200)
        img_L = self.ICRF_Map(img_rgb, index=icrf_index)
        # Step 2: Mosaic
        pattern_index = random.randint(0, 3)
        pattern = []
        if pattern_index == 0:
            pattern = 'GRBG'
        elif pattern_index == 1:
            pattern = 'RGGB'
        elif pattern_index == 2:
            pattern = 'GBRG'
        elif pattern_index == 3:
            pattern = 'BGGR'
        self.pattern = pattern
        img_mosaic = self.mosaic_bayer(img_L, pattern=pattern)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = self.add_PG_noise(img_mosaic)

        # -------- ISP PROCESS --------------------------
        # Step 2 : Demosaic
        img_demosaic = self.Demosaic(img_mosaic_noise, pattern=self.pattern)
        # Step 1 : tone mapping
        img_Irgb = self.CRF_Map(img_demosaic, index=icrf_index)

        return img_Irgb

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    isp = ISP()
    gt_path = './noise_synthetic/01_gt.png';
    gt_im = skimage.img_as_float(cv2.imread(gt_path)[:, :, ::-1]);

    pch_size = 256
    im_temp = gt_im[:pch_size, :pch_size,].astype(np.float32)
    im_noisy = isp.noise_generate_srgb(im_temp)
    plt.imshow(np.concatenate((im_noisy, im_temp), 1))
    plt.show()


