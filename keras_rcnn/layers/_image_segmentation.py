# -*- coding: utf-8 -*-

import keras.engine.topology


class ImageSegmentation(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        super(ImageSegmentation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ImageSegmentation, self).build(input_shape)

    def call(self, x, training=None, **kwargs):
        return x

    def compute_output_shape(self, input_shape):
        pass

    def compute_mask(self, inputs, mask=None):
        pass
