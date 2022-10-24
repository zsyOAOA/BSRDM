#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-04-15 19:28:31

import sys
import cv2
import math
import torch
import random
import numpy as np
from scipy import ndimage
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ResizeRight.resize_right import resize
from camera_isp.ISP_implement_cbd import ISP
from skimage import img_as_float32, img_as_ubyte

def str2bool(x):
    return x.lower() == 'true'

def jpeg_compress(im, quality=50):
    '''
    Args:
        im: numpy array with RGB channel, uint8
        quality: compression quality
    '''
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encim = cv2.imencode('.jpeg', im[:, :, ::-1], encode_param)
    decim = cv2.imdecode(encim, cv2.IMREAD_UNCHANGED)
    return decim[:, :, ::-1]

def update_args(args_json, args_parser):
    for key, value in args_json.items():
        if not key in vars(args_parser):
            setattr(args_parser, key, value)

def shifted_anisotropic_Gaussian(k_size=np.array([15, 15]),
                                 scale_factor=np.array([4, 4]),
                                 lambda_1=1.2,
                                 lambda_2=5.,
                                 theta=0,
                                 noise_level=0,
                                 shift='left'):
    """"
    # modified version of https://github.com/cszn/USRNet/blob/master/utils/utils_sisr.py
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = k_size // 2 - 0.5*(scale_factor - k_size % 2)
    elif shift.lower() == 'center':
        MU = k_size // 2
    elif shift.lower() == 'right':
        MU = k_size // 2 + 0.5*(scale_factor - k_size % 2)
    else:
        sys.exit('Please input corrected shift parameter: left, right or center!')
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # Normalize the kernel and return
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel

def imshow(x, title=None, cbar=False):
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    :parame img: uint8 or float ndarray
    '''
    in_im_type = im.dtype
    im = im.astype(np.float64)
    if in_im_type != np.uint8:
        im *= 255.
    # convert
    if only_y:
        rlt = np.dot(im, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im, np.array([[65.481,  -37.797, 112.0  ],
                                      [128.553, -74.203, -93.786],
                                      [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if in_im_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_im_type)

def ssim(im1, im2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(im1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(im2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(im1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(im2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(im1 * im2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(im1, im2, border=0, ycbcr=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    im1, im2: [0, 255]
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    if ycbcr:
        im1 = rgb2ycbcr(im1, only_y=True)
        im2 = rgb2ycbcr(im2, only_y=True)

    if im1.ndim == 2:
        return ssim(im1, im2)
    elif im1.ndim == 3:
        if im1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(im1[:,:,i], im2[:,:,i]))
            return np.array(ssims).mean()
        elif im1.shape[2] == 1:
            return ssim(np.squeeze(im1), np.squeeze(im2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0, ycbcr=False):
    '''
    im1, im2: [0, 255]
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]
    if ycbcr:
        im1 = rgb2ycbcr(im1, only_y=True)
        im2 = rgb2ycbcr(im2, only_y=True)

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def generate_gauss_kernel_mix(H, W, rng=None):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    pch_size = 32
    K_H = math.floor(H / pch_size)
    K_W = math.floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))
    if rng is None:
        centerW = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    else:
        centerW = rng.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)
    if rng is None:
        centerH = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    else:
        centerH = rng.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)
    if rng is None:
        scale = np.random.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    else:
        scale = rng.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    scale = scale.astype(np.float32)
    XX, YY = np.meshgrid(np.arange(0, W), np.arange(0,H))
    XX = XX[:, :, np.newaxis].astype(np.float32)
    YY = YY[:, :, np.newaxis].astype(np.float32)
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerW)**2-(YY-centerH)**2)/(2*scale**2) )
    out = ZZ.sum(axis=2, keepdims=False) / K

    return out

def degradation(im_HR, sf, kernel, noise_level,
                convolve='False',
                noise_type='signal',
                downsampler='direct',
                padding_mode='mirror'):
    '''
    Args:
        im_HR: numpy array, h x w x c, [0, 1], float
        sf: int
        kernel: k x k, float
        noise_level: float or 2-length tuple or list
        convolve: bool, if Ture convolve else correlate
        noise_type: str, 'Gaussin', 'Mix', 'jpeg' or 'signal'
        downsample: str, 'direct' or 'bicubic'
        padding_mode: str, 'mirror' or 'warp'
    '''
    if im_HR.ndim == 2:
        im_HR = im_HR[:, :, None]
    im_HR = modcrop(im_HR, sf)
    # blur
    if convolve:
        im_blur = ndimage.convolve(im_HR, kernel[:, :, None], mode=padding_mode)
    else:
        im_blur = ndimage.correlate(im_HR, kernel[:, :, None], mode=padding_mode)
    # downsampling
    if downsampler.lower() == 'direct':
        im_blur = im_blur[0::sf, 0::sf, ]
    elif downsampler.lower() == 'bicubic':
        im_blur = resize(im_blur, scale_factors=1/sf)
    else:
        sys.exit('Please input the corrected downsample type: Direct or bicubic')

    # adding noise
    # if noise_type.lower() == 'mix':
        # h, w = im_blur.shape[:2]
        # var_map = generate_gauss_kernel_mix(256, 256)
        # var_map = cv2.resize(var_map, (w, h), interpolation=cv2.INTER_LINEAR)
        # var_map = (0.1/255.) + (var_map - var_map.min()) / (var_map.max()-var_map.min()) * ((noise_level - 0.1)/255.)
        # im_LR = im_blur + np.random.randn(*im_blur.shape) * var_map[:, :, None]
    if noise_type.lower() == 'signal':
        isp = ISP()
        im_LR = isp.noise_generate_srgb(im_blur)
    elif noise_type.lower() == 'gaussian':
        im_LR = im_blur + np.random.randn(*im_blur.shape) * (noise_level/255.)
    else:
        sys.exit('Please input corrected noise type: Gaussian, Mix and Peaks!')

    return im_LR.squeeze()

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def set_seed(seed):
    # print('Setting random seed: {:d}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ImPad:
    '''
    Make dimensions divisible by `d`
    '''
    def __init__(self, im, d):
        '''
        im: ndarry, h x w x c
        d: integer
        '''
        old_h, old_w = im.shape[:2]
        self.new_h, self.new_w = d * math.ceil(old_h / d), d * math.ceil(old_w / d)
        pad_h, pad_w = self.new_h - old_h, self.new_w - old_w
        self.padding_h = (int(pad_h / 2), pad_h - int(pad_h / 2))
        self.padding_w = (int(pad_w / 2), pad_w - int(pad_w / 2))
        self.padding = (self.padding_h, self.padding_w, (0, 0))
        self.im_old = im.copy()

    def pad_fun(self):
        im_new = np.pad(self.im_old, pad_width=self.padding, mode='reflect')

        return im_new

    def pad_inverse(self, im_target):
        return im_target[self.padding_h[0]:self.new_h-self.padding_h[1],
                         self.padding_w[0]:self.new_w-self.padding_w[1]]

def center_crop(im, psize=1024):
    '''
    Crop a psize x psize patch from the original image.
    Input:
        im: h x w x c or h x w image
    '''
    h, w = im.shape[:2]
    if h > psize:
        h_start = (h - psize) // 2
        h_end = h_start - (h - psize)
        im = im[h_start:h_end,]
    if w > psize:
        w_start = (w - psize) // 2
        w_end = w_start - (w - psize)
        im = im[:, w_start:w_end,]
    return im

def modcrop(im, sf):
    '''
    Args:
        im: numpy array, h x w x c
        sf: scale factor
    '''
    h, w = im.shape[:2]
    return im[:h - h % sf, :w - w % sf, ]


