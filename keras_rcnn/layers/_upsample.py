# -*- coding: utf-8 -*-

import keras.engine.topology

import keras_rcnn.backend


class Upsample(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        super(Upsample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Upsample, self).build(input_shape)

    def call(self, inputs, **kwargs):
        output, target = inputs

        shape = keras.backend.shape(target)

        return keras_rcnn.backend.resize(output, (shape[1], shape[2]))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[1][1:3]

        return (input_shape[0][0],) + output_shape + (input_shape[0][-1],)
