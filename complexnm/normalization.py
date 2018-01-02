#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Time      : 2017/12/29 16:27
# Author    : zsh_o

import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K

class ComplexBatchNormalization(Layer):
    # keras官方batchnormalization在这里加了一个broadcasting，不是很明白为啥要加，就没加
    def __init__(self,
                 axis = -1,
                 momentum = 0.9,
                 epsilon = 1e-4,
                 center = True,
                 scale = True,
                 beta_initializer = 'zeros',
                 gamma_diag_initializer = 'sqrt_init', # gamma is matrix with freedom of three degrees: rr, ri, ii
                 gamma_off_initializer = 'zeros',
                 moving_mean_initializer = 'zeros',

                 ):