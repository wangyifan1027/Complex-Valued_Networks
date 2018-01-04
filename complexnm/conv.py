#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Time      : 2018/1/4 9:01
# Author    : zsh_o

from keras import backend as K
from keras import activations, regularizers, constraints
from . import initializers as cinitializers
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils
import numpy as np

class _ComplexConv(Layer):
    # 论文源代码中这个地方外加了两个功能，都是针对权值的
    #   1、spectral parametrization: 表示权值是在谱域定义的，进行卷积之前需要把其用IFFT转变为原始的空域
    #   2、normalize weight: 每一次进行卷积之前都把权值归一化，采用的是bn里面定义的complex_normalization，这也应该是在complexBN中作者没有把那两个函数放到类里面的原因吧
    # 关于1，和pooling里面的那个SpectralPooling均是采用论文：Spectral Representations for Convolutional Neural Networks（https://arxiv.org/abs/1506.03767）
    #       卷积的权值在频率域表示，进行卷积之前需要用IFFT把其反变换为空域，论文中显示，这种表示方法学习到的权值更稀疏更少（学习改变的权值的量少）
    # 关于2，个人感觉没什么用，在这个地方直接对权值进行操作不如加相似功能的正则化
    pass