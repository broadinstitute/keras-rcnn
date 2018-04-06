# -*- coding: utf-8 -*-

import keras.engine.topology

import keras_rcnn.backend


class GradientReversal(keras.engine.topology.Layer):
    def __init__(self, hp_lambda, **kwargs):
        self.hp_lambda = hp_lambda

        super(GradientReversal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GradientReversal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return keras_rcnn.backend.reverse_gradient(inputs, self.hp_lambda)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        configuration = {
            "hp_lambda": self.hp_lambda
        }

        return {**super(GradientReversal, self).get_config(), **configuration}
