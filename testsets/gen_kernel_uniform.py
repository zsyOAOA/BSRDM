#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-04-15 19:40:10

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import scipy.io as sio
from utils_bsrdm import shifted_anisotropic_Gaussian

num_kernel = 6
for s in [2, 3, 4]: 
    p = (s * 2 + 1) * 2 + 1
    kernels = np.zeros([p, p, num_kernel])
    kernels[:, :, 0] = shifted_anisotropic_Gaussian(np.array([p, p]), np.array([s, s]), 1.6, 1.6, 0, 0)
    kernels[:, :, 1] = shifted_anisotropic_Gaussian(np.array([p, p]), np.array([s, s]), 2.0, 2.0, 0, 0)
    kernels[:, :, 2] = shifted_anisotropic_Gaussian(np.array([p, p]), np.array([s, s]), s*1.5, s*0.75, np.pi*0, 0)
    kernels[:, :, 3] = shifted_anisotropic_Gaussian(np.array([p, p]), np.array([s, s]), s*1.5, s*0.50, np.pi*0.75, 0)
    kernels[:, :, 4] = shifted_anisotropic_Gaussian(np.array([p, p]), np.array([s, s]), s*1.5, s*0.50, np.pi*0.25, 0)
    kernels[:, :, 5] = shifted_anisotropic_Gaussian(np.array([p, p]), np.array([s, s]), s*1.5, s*0.75, np.pi*0.50, 0)

    kernel_dir = Path('./testsets/kernels_synthetic')
    if not kernel_dir.exists():
        kernel_dir.mkdir(parents=True)
    sio.savemat(str(kernel_dir / ('kernels' + '_x' + str(s) +'.mat')), {'kernels':kernels})

