#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-04-16 16:50:36

import os
import sys
import random
import numpy as np
from SSIM import SSIM
from math import sqrt, ceil, inf
from shutil import rmtree
from pathlib import Path
from network.skip import skip
from skimage import img_as_float32, img_as_ubyte

import utils_bsrdm as utils
from ResizeRight.resize_right import resize

import torch
from torch import optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.nn.utils as nutils
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type(torch.FloatTensor)

def ekp_kernel_generator(U, kernel_size=15, sf=3, shift='left'):
    '''
    Generate Gaussian kernel according to cholesky decomposion.
    \Sigma = M * M^T, M is a lower triangular matrix.
    Input:
        U: 2 x 2 torch tensor
        sf: scale factor
    Output:
        kernel: 2 x 2 torch tensor
    '''
    #  Mask
    mask = torch.tensor([[1.0, 0.0],
                         [1.0, 1.0]], dtype=torch.float32).to(U.device)
    M = U * mask

    # Set COV matrix using Lambdas and Theta
    INV_SIGMA = torch.mm(M.t(), M)

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = kernel_size // 2 - 0.5 * (sf - 1)
    elif shift.lower() == 'center':
        MU = kernel_size // 2
    elif shift.lower() == 'right':
        MU = kernel_size // 2 + 0.5 * (sf - 1)
    else:
        sys.exit('Please input corrected shift parameter: left , right or center!')

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    Z = torch.stack((X, Y), dim=2).unsqueeze(3).to(U.device)   # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.permute(0,1,3,2)                  # k x k x 1 x 2
    raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))

    # Normalize the kernel and return
    kernel = raw_kernel / torch.sum(raw_kernel)   # k x k
    return kernel.unsqueeze(0).unsqueeze(0)

class Trainer:
    def __init__(self, args, im_LR, im_HR=None, kernel_gt=None):
        '''
        Args:
            im_LR: ndarray, h x w x c, [0, 1.], float
            im_HR: ndarray, h x w x c, [0, 1.], float or [0, 255] uint8
        '''
        self.args = args

        # setting GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

        # load data
        self.tidy_data_LR(im_LR)              # self.im_LR: 1 x c x h x w, torch tensor, float32, [0, 1], cuda
        self.tidy_data_HR(im_HR, kernel_gt)   # self.im_HR: h x w x c, numpy array, uint8, [0, 255]
                                              # self.kernel_gt: 1 x 1 x k x k, torch tensor, float32
        self.num_pixels = self.im_LR.numel()

        # determine kernel size
        self.kernel_size = (args.sf * 2 + 1) * 2 + 1

        # make gradient filters
        self.make_gradient_filter()   # self.grad_filters: 4 x 3 x 3, torch tensor, cuda

        # make average filter to estimate variance
        self.make_variance_filter()   # self.grad_filters: 1 x 1 x p x p, torch tensor, cuda

    def tidy_data_LR(self, im_LR):
        if im_LR.ndim == 3:
            im_LR_torch = torch.from_numpy(im_LR.transpose([2,0,1])).type(torch.float32)
        elif im_LR.ndim == 2:
            im_LR_torch = torch.from_numpy(im_LR[None, :, :]).type(torch.float32)
        else:
            sys.exit('The loaded image must have 3 (Color) or 2 (Gray) dimensions!')

        self.im_LR = im_LR_torch.unsqueeze(0).cuda()

    def tidy_data_HR(self, im_HR=None, kernel_gt=None):
        if im_HR is None:
            self.im_HR = None
        else:
            im_HR = utils.modcrop(im_HR, self.args.sf)
            self.im_HR = img_as_ubyte(im_HR) if im_HR.dtype != np.uint8 else im_HR

            if kernel_gt is not None:
                self.kernel_gt = torch.from_numpy(kernel_gt[None, None, ]).type(torch.float32).cuda()

    def open_log(self):
        self.log_loss_step = 0
        self.log_im_step = 0
        if Path(self.args.log_dir).is_dir():
            rmtree(str(Path(self.args.log_dir)))
        Path(self.args.log_dir).mkdir()

        self.writer = SummaryWriter(str(Path(self.args.log_dir)))

    def close_log(self):
        self.writer.close()

    def make_gradient_filter(self):
        filters = np.zeros([4, 3, 3], dtype=np.float32)
        filters[0,] = np.array([[0,  -1, 0],
                                [0,  1,  0],
                                [0,  0,  0]])

        filters[1,] = np.array([[-1, 0,  0],
                                [0,  1,  0],
                                [0,  0,  0]])

        filters[2,] = np.array([[0,  0, 0],
                                [-1, 1, 0],
                                [0,  0, 0]])

        filters[3,] = np.array([[0,  0, 0],
                                [0,  1, 0],
                                [-1, 0, 0]])

        self.grad_filters = torch.from_numpy(filters).cuda()

    def make_variance_filter(self):
        filter_average = np.ones((self.args.window_variance,)*2) / (self.args.window_variance ** 2)
        self.var_filter = torch.from_numpy(filter_average[None, None, ]).type(torch.float32).cuda()

    def blur_and_downsample(self,  padding_mode="reflect"):
        '''
        Args:
            im_HR: N x c x h x w, torch tensor
            kernel: 1 x 1 x k x k, torch tensor
        '''
        hr_pad = F.pad(input = self.im_HR_est, mode = padding_mode, pad = (self.kernel_size // 2, ) * 4)
        out = F.conv2d(hr_pad, self.kernel_est.expand(hr_pad.shape[1], -1, -1, -1), groups=hr_pad.shape[1])
        if self.args.downsampler.lower() == 'direct':
            self.im_LR_est = out[:, :, 0::self.args.sf, 0::self.args.sf]
        elif self.args.downsampler.lower() == 'bicubic':
            self.im_LR_est = resize(out, scale_factors=1/self.args.sf)
        else:
            sys.exit('Please input corrected downsampler: Bicubic or Downsampler!')

        return self.im_LR_est

    def calculate_grad_abs(self, padding_mode="reflect"):
        hr_pad = F.pad(input = self.im_HR_est, mode = padding_mode, pad = (1, ) * 4)
        out = F.conv3d(input = hr_pad.expand(self.grad_filters.shape[0], -1, -1, -1).unsqueeze(0),
                       weight = self.grad_filters.unsqueeze(1).unsqueeze(1),
                       stride = 1, groups = self.grad_filters.shape[0])

        return torch.abs(out.squeeze(0))

    def estimate_variance(self, padding_mode="reflect"):
        noise2 = (self.im_LR - self.im_LR_est.data)**2
        if self.args.noise_estimator.lower() == 'niid':
            noise2_pad = F.pad(input=noise2, mode = padding_mode, pad = ((self.args.window_variance - 1) //2, )*4)
            self.lambda_p = F.conv2d(input = noise2_pad,
                                     weight = self.var_filter.expand(self.im_LR.shape[1], -1, -1, -1),
                                     groups= self.im_LR.shape[1])
        elif self.args.noise_estimator.lower() == 'iid':
            self.lambda_p = torch.ones_like(self.im_LR) * noise2.mean()
        else:
            sys.exit('Please input corrected noise estimation methods: iid or niid!')

    def get_loss_Mstep(self, eps=1e-8):
        likelihood = 0.5 * ((1/self.lambda_p) * (self.im_LR - self.blur_and_downsample())**2).sum() / self.num_pixels
        # adding 1e-8 to avoid nan in backward
        grad_loss = self.args.rho * torch.pow(self.calculate_grad_abs()+eps, self.args.gamma).sum() / self.num_pixels
        loss = likelihood + grad_loss + 0.5 * (self.kernel_code**2).sum() / self.num_pixels

        return loss, likelihood,  grad_loss

    def get_loss_Estep(self, eps=1e-8):
        likelihood = 0.5 * ((1/self.lambda_p) * (self.im_LR - self.blur_and_downsample())**2).sum()
        grad_loss = self.args.rho * torch.pow(self.calculate_grad_abs()+eps, self.args.gamma).sum()
        loss = likelihood + grad_loss + 0.5 * (self.Z**2).sum()

        return loss

    def freeze_parameters(self):
        for p in self.generator.parameters():
            p.requires_grad = False
        self.Z.requires_grad = True

    def unfreeze_parameters(self):
        for p in self.generator.parameters():
            p.requires_grad = True
        self.Z.requires_grad = False

    def calculate_metrics(self, ycbcr=True):
        im_HR_est = img_as_ubyte(np.clip(self.im_HR_est.data.cpu().numpy().transpose([0,2,3,1]).squeeze(), 0.0, 1.0))
        if self.im_HR.ndim == 2:
            ycbcr = False
        psnr = utils.calculate_psnr(self.im_HR, im_HR_est, border=self.args.sf**2, ycbcr=ycbcr)
        ssim = utils.calculate_ssim(self.im_HR, im_HR_est, border=self.args.sf**2, ycbcr=ycbcr)

        return psnr, ssim

    def clip_grad_Z(self):
        self.norm_Z = torch.norm(self.Z.grad.detach(), 2)
        clip_coef = 1 / (self.norm_Z + 1e-6)
        self.Z.grad.detach().mul_(clip_coef)

    def initialize_G(self, lr_G_temp=2e-3):
        self.generator = skip(num_input_channels = self.args.input_chn,
                              num_output_channels = self.im_LR.shape[1],
                              num_channels_down = self.args.down_chn_G,
                              num_channels_up = self.args.up_chn_G,
                              num_channels_skip = self.args.skip_chn_G,
                              upsample_mode='bilinear',
                              need_sigmoid=True,
                              need_bias=True,
                              pad='reflection',
                              act_fun='LeakyReLU',
                              use_bn = self.args.use_bn_G).cuda()
        self.optimizer_G = optim.Adam(params=self.generator.parameters(), lr=lr_G_temp)
        if self.args.disp == 1:
            log_str = 'Number of parameters in Generator: {:.2f}K'
            print(log_str.format(utils.calculate_parameters(self.generator)/1000))
            print('Initiliazing the generator...')

        H_up, W_up = int(self.im_LR.shape[2] * self.args.sf), int(self.im_LR.shape[3] * self.args.sf)
        self.Z = torch.randn([1, self.args.input_chn, H_up, W_up]).cuda()
        self.lambda_p = torch.ones_like(self.im_LR, requires_grad=False) * (0.01**2)

        ssimloss = SSIM()
        im_HR_base = F.interpolate(self.im_LR, size=(H_up, W_up), mode='bilinear', align_corners=False)
        for kk in range(50):
            self.optimizer_G.zero_grad()
            self.im_HR_est = self.generator(self.Z)
            loss = 1 - ssimloss(im_HR_base, self.im_HR_est)
            loss.backward()
            self.optimizer_G.step()

        self.optimizer_G.param_groups[0]['lr'] = self.args.lr_G

    def initialize_K(self):
        if self.args.disp == 1:
            print('Initiliazing the kernel...')

        # Kernel Prior
        l1 = 1 / (self.args.sf * 1.00)
        self.kernel_code = torch.tensor([[l1,  0.0],
                                         [0.0, l1]], dtype=torch.float32).cuda()
        self.kernel_code.requires_grad = True
        self.optimizer_kernel = optim.Adam(params=[self.kernel_code,], lr=self.args.lr_K)

    def get_HR_res(self):
        return img_as_ubyte(self.im_HR_est.detach().cpu().squeeze(0).clamp_(0.0,1.0).numpy().transpose((1,2,0)))

    def get_kernel_est(self):
        return self.kernel_est.detach().cpu().squeeze().numpy()

    def train(self):
        # print options
        if self.args.disp == 1:
            for key in vars(self.args):
                value = str(getattr(self.args, key))
                print('{:25s}: {:s}'.format(key, value))
            self.open_log()

        self.initialize_G()

        self.initialize_K()

        # begin training
        num_iters = 0
        for ii in range(ceil(self.args.max_iters / self.args.internal_iter_M)):
            # M-Step: update generator parameter and kernel code
            for jj in range(self.args.internal_iter_M):
                num_iters += 1
                self.optimizer_G.zero_grad()
                self.optimizer_kernel.zero_grad()
                self.im_HR_est = self.generator(self.Z)
                self.kernel_est = ekp_kernel_generator(self.kernel_code,
                                                       kernel_size=self.kernel_size,
                                                       sf=self.args.sf,
                                                       shift=self.args.kernel_shift)
                loss, likelihood, grad_loss = self.get_loss_Mstep()
                loss.backward()
                grad_norm_G = nutils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm_G)
                self.optimizer_kernel.step()
                self.optimizer_G.step()

                if num_iters % self.args.print_freq == 0 and self.args.disp == 1:
                    lr_G = self.optimizer_G.param_groups[0]['lr']
                    lr_K = self.optimizer_kernel.param_groups[0]['lr']
                    if self.im_HR is None:
                        log_str = 'Iter:{:04d}/{:04d}, Loss:{:.2e}/{:.2e}/{:.2e}, normG:{:.2e}/{:.2e}, lrG:{:.2e}/{:.2e}'
                        print(log_str.format(num_iters, self.args.max_iters, loss.item(), likelihood.item(),
                                                   grad_loss.item(), grad_norm_G, self.args.max_grad_norm_G, lr_G, lr_K))
                    else:
                        psnr, ssim = self.calculate_metrics(ycbcr=True)
                        log_str = 'Iter:{:04d}/{:04d}, Loss:{:5.3f}/{:5.3f}/{:5.3f}, PSNR:{:5.2f}, ' + \
                                                                 'SSIM:{:6.4f}, normG:{:.2e}/{:.2e}, lrG/K:{:.2e}/{:.2e}'
                        print(log_str.format(num_iters, self.args.max_iters, loss.item(), likelihood.item(),
                                       grad_loss.item(), psnr, ssim, grad_norm_G, self.args.max_grad_norm_G, lr_G, lr_K))

                    # tensorboard
                    self.writer.add_scalar('LossM', loss.item(), self.log_loss_step)
                    self.log_loss_step += 1
                    if self.im_HR is None:
                        x1 = vutils.make_grid(self.im_HR_est)
                        self.writer.add_image('Estimated HR Images', x1, self.log_im_step)
                        x2 = vutils.make_grid(self.kernel_est)
                        self.writer.add_image('Estimated Kernel', x2, self.log_im_step)
                    else:
                        im_HR_temp = self.im_HR[:, :, None] if self.im_HR.ndim == 2 else self.im_HR
                        x1 = vutils.make_grid(torch.cat((
                            torch.from_numpy(im_HR_temp.transpose([2,0,1])[None,].copy()).type(torch.float32).cuda()/255.,
                            self.im_HR_est), 0))
                        self.writer.add_image('GT and Estimated HR Images', x1, self.log_im_step)
                        x2 = vutils.make_grid(torch.cat((self.kernel_gt, self.kernel_est.data), 0), normalize=True, scale_each=True)
                        self.writer.add_image('GT and Estimated Kernel', x2, self.log_im_step)
                    x3 = vutils.make_grid(self.im_LR, 0)
                    self.writer.add_image('LR Image', x3, self.log_im_step)
                    self.log_im_step += 1

            # update noise variance
            if num_iters < 300:
                self.estimate_variance()

            # E-Step
            self.freeze_parameters()
            with torch.set_grad_enabled(False):
                self.kernel_est = ekp_kernel_generator(self.kernel_code,
                                                       kernel_size=self.kernel_size,
                                                       sf=self.args.sf,
                                                       shift=self.args.kernel_shift)
            for kk in range(self.args.langevin_steps):
                self.im_HR_est = self.generator(self.Z)
                loss = self.get_loss_Estep()

                loss.backward()
                self.clip_grad_Z()

                self.Z.data = self.Z.data - 0.5 * self.args.delta**2 * self.Z.grad.data
                if kk < (self.args.langevin_steps / 3.0):
                    self.Z.data = self.Z.data + self.args.delta * torch.randn_like(self.Z) / self.norm_Z
                self.Z.grad.fill_(0)
            self.unfreeze_parameters()

        if self.args.disp == 1:
            self.close_log()

