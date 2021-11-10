#!/usr/bin/env python
# -*- coding:utf-8 -*-

from torch import nn
from .common import *
from .non_local_dot_product import NONLocalBlock2D

def skip(num_input_channels=2,
         num_output_channels=3,
         n_scales = 3,
         num_channels_down=[64, 64, 64],
         num_channels_up=[64, 64, 64],
         num_channels_skip=[16, 16, 16],
         filter_size_down=3, filter_size_up=3, filter_skip_size=1,
         need_sigmoid=True, need_bias=True,
         pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
         need1x1_up=True, use_bn=True):
    """
    Network of DIP: Assembles encoder-decoder with skip connections.
    Reference: https://github.com/DmitryUlyanov/deep-image-prior 

    The non-local block is borrowed from DIPFKP.
    Reference: https://github.com/JingyunLiang/FKP

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    if isinstance(num_channels_down, int):
        num_channels_down = [num_channels_down, ] * n_scales
    else:
        assert n_scales == len(num_channels_down)
    
    if isinstance(num_channels_up, int):
        num_channels_up = [num_channels_up, ] * n_scales
    else:
        assert n_scales == len(num_channels_up)

    if isinstance(num_channels_skip, int):
        num_channels_skip = [num_channels_skip, ] * n_scales
    else:
        assert n_scales == len(num_channels_down)

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()  # this is a pointer
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()  # encoder; decoder is directly in model_tmp
        skip = nn.Sequential()  # skip

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))  # cat encoded feature and shortcut's along channel
        else:
            model_tmp.add(deeper)

        if use_bn:
            model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:  # skip=conv+bn+act
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            if use_bn:
                skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        if use_bn:
            deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        # only difference from original
        if i > 1:
            deeper.add(NONLocalBlock2D(in_channels=num_channels_down[i]))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        if use_bn:
            deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest, directly cat deeper and skip
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)  # add an entrance for next level, will be save by model_tmp
            k = num_channels_up[i + 1]  # the output from next-level decoder

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=False))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias,
                           pad=pad))  # encoder+skip
        if use_bn:
            model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            if use_bn:
                model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
