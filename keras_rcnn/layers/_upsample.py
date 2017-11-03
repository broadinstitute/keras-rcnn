# -*- coding: utf-8 -*-

import keras.engine.topology

import keras_rcnn.backend


class Upsample(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        super(Upsample, self).__init__(**kwargs)

    def build(self, input_shape):

        super(Upsample, self).build(input_shape)

    def call(self, inputs, **kwargs):
        image, target = inputs

        shape = keras.backend.shape(target)

        output_shape = (shape[1], shape[2])

        return keras_rcnn.backend.resize(image, output_shape)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]

        output_shape = input_shape[1][1:3]

        channels = input_shape[0][-1]

        return (batch_size,) + output_shape + (channels,)
