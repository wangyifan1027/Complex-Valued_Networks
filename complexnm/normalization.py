#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Time      : 2017/12/29 16:27
# Author    : zsh_o

import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K

def sqrt(shape, dtype = None):
    value = (1 / K.sqrt(2)) * K.ones(shape)
    return value

def initGet(init):
    if init in ['sqrt']:
        return sqrt
    else:
        return initializers.get(init)

def initSet(init):
    if init in [sqrt]:
        return 'sqrt'
    else:
        return initializers.serialize(init)

class ComplexBatchNormalization(Layer):
    # keras官方batchnormalization在这里加了一个broadcasting，不是很明白为啥要加，就没加
    # 把计算inference的所有函数转到class里面
    # 复数相关的层，复数的输入输出为两个张量，real part and image part, 对应的input and output shape 也为两个
    def __init__(self,
                 axis = -1,
                 momentum = 0.9,
                 epsilon = 1e-4,
                 center = True,
                 scale = True,
                 beta_initializer = 'zeros',
                 gamma_diag_initializer = 'sqrt_init', # gamma is matrix with freedom of three degrees: rr, ri, ii
                 gamma_off_initializer = 'zeros',
                 moving_mean_initializer = 'zeros', # 三个moving_average变量均不可训练，用于计算和保存均值和协方差矩阵
                 moving_variance_initializer = 'sqrt_init', # 每次计算该batch的均值和协方差矩阵，然后用加动量的moving_average更新moving_mean, moving_var, moving_cov
                 moving_covariance_initializer = 'zeros',
                 beta_regularizer = None,
                 gamma_diag_regularizer = None, # 正则化
                 gamma_off_regularizer = None,
                 beta_constraint = None, # 约束
                 gamma_diag_constraint = None,
                 gamma_off_constraint = None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer               = initGet(beta_initializer)
        self.gamma_diag_initializer         = initGet(gamma_diag_initializer)
        self.gamma_off_initializer          = initGet(gamma_off_initializer)
        self.moving_mean_initializer        = initGet(moving_mean_initializer)
        self.moving_variance_initializer    = initGet(moving_variance_initializer)
        self.moving_covariance_initializer  = initGet(moving_covariance_initializer)
        self.beta_regularizer               = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer         = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer          = regularizers.get(gamma_off_regularizer)
        self.beta_constraint                = constraints.get(beta_constraint)
        self.gamma_diag_constraint          = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint           = constraints.get(gamma_off_constraint)

    def build(self, input_shape):
        input_shapes = input_shape
        assert(input_shapes[0] == input_shapes[1])
        input_shape = input_shapes[0]
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim = len(input_shape),axes = {self.axis: dim})
        shape = (dim,) # 用于对实数的初始化
        if self.scale:
            self.gamma_rr = self.add_weight(shape = shape,
                                            name = 'gamma_rr',
                                            initializer = self.gamma_diag_initializer,
                                            regularizer = self.gamma_diag_regularizer,
                                            constraint = self.gamma_diag_constraint)
            self.gamma_rr = self.add_weight(shape = shape,
                                            name = 'gamma_ii',
                                            initializer = self.gamma_diag_initializer,
                                            regularizer = self.gamma_diag_regularizer,
                                            constraint = self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape = shape,
                                            name = 'gamma_ri',
                                            initializer = self.gamma_off_initializer,
                                            regularizer = self.gamma_off_regularizer,
                                            constraint = self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape = shape,
                                              initializer = self.moving_variance_initializer,
                                              name = 'moving_Vrr',
                                              trainable = False)
            self.moving_Vii = self.add_weight(shape = shape,
                                              initializer = self.moving_variance_initializer,
                                              name = 'moving_Vii',
                                              trainable = False)
            self.moving_Vri = self.add_weight(shape = shape,
                                              initializer = self.moving_covariance_initializer,
                                              name = 'moving_Vri',
                                              trainable = False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta_real = self.add_weight(shape = shape,
                                        name = 'beta_real',
                                        initializer = self.beta_initializer,
                                        regularizer = self.beta_regularizer,
                                        constraint = self.beta_constraint)
            self.beta_image = self.add_weight(shape = shape,
                                             name = 'beta_image',
                                             initializer = self.beta_initializer,
                                             regularizer = self.beta_regularizer,
                                             constraint = self.beta_constraint)
            self.moving_mean_real = self.add_weight(shape = shape,
                                               initializer = self.moving_mean_initializer,
                                               name = 'moving_mean_real',
                                               trainable = False)
            self.moving_mean_image = self.add_weight(shape = shape,
                                                    initializer = self.moving_mean_initializer,
                                                    name = 'moving_mean_image',
                                                    trainable = False)
        else:
            self.beta_real = None
            self.beta_image = None
            self.moving_mean_real = None
            self.moving_mean_image = None

        self.built = True

    def call(self, inputs, training = None):
        assert isinstance(inputs, list)
        input_real, input_image = inputs[0], inputs[1]
        input_shape = K.int_shape(input_real)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim)) # 计算均值的时候需要指定维度
        del reduction_axes[self.axis]
        mu_real = K.mean(input_real, axis = reduction_axes) # 复数加减不涉及实虚部转换
        mu_image = K.mean(input_image, axis = reduction_axes)

        # center_x = x - E[x]
        if self.center:
            centered_real = input_real - mu_real
            centered_image = input_image - mu_image
        else:
            centered_real = input_real
            centered_image = input_image
        centered_squared_real = centered_real ** 2
        centered_squared_image = centered_image ** 2
        centered = K.concatenate([centered_real, centered_image])
        centered_squared = K.concatenate([centered_squared_real, centered_squared_image])

        if self.scale:
            Vrr = K.mean( # Vrr = Cov(R(x), R(x)), Cov(X, Y) = E((X - E(X))(Y - E(Y)))
                centered_squared_real,
                axis = reduction_axes,
            ) + self.epsilon
            Vii = K.mean( # Vii = Cov(I(x), I(x))
                centered_squared_image,
                axis = reduction_axes,
            ) + self.epsilon
            Vri = K.mean( # Vri = Cov(R(x), I(x))
                centered_real * centered_image,
                axis = reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')
        input_bn = self.complexBN(centered_real, centered_image, Vrr, Vii, Vri)
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean_real, mu_real, self.momentum))
                update_list.append(K.moving_average_update(self.moving_mean_image, mu_image, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
                #----------------

    def complexBN(self, centered_real, centered_image, Vrr, Vii, Vri):
        output_real = centered_real
        output_image = centered_image
        if self.scale:
            t_real,t_image = self.complex_std(centered_real, centered_image, Vrr, Vii, Vri)
            output_real = self.gamma_rr * t_real + self.gamma_ri * t_image
            output_image = self.gamma_ri * t_real + self.gamma_ii * t_image
        if self.center:
            output_real = output_real + self.beta_real
            output_image = output_image + self.beta_image

        return output_real, output_image

    def complex_std(self, centered_real, centered_image, Vrr, Vii, Vri):
        # sqrt of a 2x2 matrix, https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = np.sqrt(delta)
        t = np.sqrt(tau + 2 * s)

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii +s) * inverse_st
        Wii = (Vrr +s) * inverse_st
        Wri = -Vri * inverse_st

        output_real = Wrr * centered_real + Wri * centered_image
        output_image = Wri * centered_real + Wii * centered_image

        return output_real, output_image


















