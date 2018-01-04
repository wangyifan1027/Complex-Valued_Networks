#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Time      : 2018/1/4 14:43
# Author    : zsh_o

from keras import backend as K
from keras.layers import Layer, InputSpec
import numpy as np
import tensorflow as tf

class _SpectralPooling(Layer):
    # SpectralPooling based on FFT
    # 首先进行傅立叶变换FFT转换到频率域，然后从低频开始截取其中连续的一部分的频段信息，其余置0，再经过反傅立叶变换转回空域
    # tensorflow的fft函数有问题，fft的执行速度应该跟卷积不相伯仲才对，但这里tensorflow的fft很慢，而且只能对batch中的单张图片进行FFT，也就是说每一个要执行filters次fft和ifft，这样的话，可以用卷积来改写FFT和IFFT
    # tensorflow的fft没有经过中心移位操作，只需在每个维度上截取相应的前几位即可，采用tf.split来完成
    def __init__(self, rank, gamma, topf, **kwargs):
        # 可选参数，gamma或者topf，代表保留的量
        # gamma：保留前百分比的信息
        # topf：保留前topf个分量
        super(_SpectralPooling, self).__init__(**kwargs)
        assert gamma is not None or topf is not None
        if gamma is not None:
            self.gamma = get_tuple(gamma, rank)
            self.topf = None
        else:
            self.topf = get_tuple(topf, rank)
            self.gamma = None
        self.fftFunc = {
            1:  tf.fft,
            2:  tf.fft2d,
            3:  tf.fft3d
        }[rank]
        self.ifftFunc = {
            1:  tf.ifft,
            2:  tf.ifft2d,
            3:  tf.ifft3d
        }[rank]
        self.rank = rank
    def call(self, inputs, mask = None):
        assert isinstance(inputs, list)
        input_real, input_image = inputs[0], inputs[1]
        input_shape = K.int_shape(input_real)
        input = tf.complex(input_real, input_image)
        t = tf.unstack(input, axis = -1)
        spectral_input = [tf.expand_dims(self.fftFunc(k), -1) for k in t]
        spectral_input = tf.concat(spectral_input, -1)
        # 截取操作 [batch, (data_shape), filters], len(data_shape) = rank
        for rindex in range(self.rank):
            # 把第rindex维分裂成两部分
            if self.gamma is not None:
                itopf = self.gamma[rindex] * input_shape[rindex + 1]
                if itopf < 1:
                    itopf = 1
            else:
                itopf = self.topf[rindex]
            rif = input_shape[rindex + 1] - itopf
            # itopf 需要保留的个数，rif 需要置零的个数
            sp = tf.split(spectral_input, [itopf, input_shape[rindex + 1] - itopf], axis = rindex + 1)
            sp[1] *= 0. # 剩余部分置零
            spectral_input = tf.concat(sp, axis = rindex +1)

        output = [tf.expand_dims(self.ifftFunc(k), -1) for k in spectral_input]
        output = tf.concat(output, -1)
        output_real = tf.real(output)
        output_image = tf.image(output)
        return output_real, output_image

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        assert (input_shapes[0] == input_shapes[1])
        return input_shapes

    def get_config(self):
        config = {
            'rank': self.rank,
            'gamma': self.gamma,
            'topf': self.topf
        }
        base_config = super(_SpectralPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_tuple(value, n):
    if isinstance(value, float):
        return (value,)*n
    elif isinstance(value, tuple):
        if len(value) == n:
            return value
        else:
            raise ValueError('' + str(value) + 'is invalid.')
    else:
        raise ValueError('' + str(value) + 'is invalid.')


class SpectralPooling1D(_SpectralPooling):
    def __init__(self, gamma, topf, **kwargs):
        super(SpectralPooling1D, self).__init__(rank = 1,
                                                gamma = gamma,
                                                topf = topf,
                                                **kwargs)

    def get_config(self):
        config = super(SpectralPooling1D, self).get_config()
        config.pop('rank')
        return config


class SpectralPooling2D(_SpectralPooling):
    def __init__(self, gamma, topf, **kwargs):
        super(SpectralPooling2D, self).__init__(rank = 2,
                                                gamma = gamma,
                                                topf = topf,
                                                **kwargs)

    def get_config(self):
        config = super(SpectralPooling2D, self).get_config()
        config.pop('rank')
        return config


class SpectralPooling3D(_SpectralPooling):
    def __init__(self, gamma, topf, **kwargs):
        super(SpectralPooling3D, self).__init__(rank = 3,
                                                gamma = gamma,
                                                topf = topf,
                                                **kwargs)

    def get_config(self):
        config = super(SpectralPooling3D, self).get_config()
        config.pop('rank')
        return config