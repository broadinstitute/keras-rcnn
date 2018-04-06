# -*- coding: utf-8 -*-

import keras.engine.topology


class ImageAdaptor(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        super(ImageAdaptor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ImageAdaptor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        configuration = {}

        return {
            **super(ImageAdaptor, self).get_config(),
            **configuration
        }


class InstanceAdaptor(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        super(InstanceAdaptor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InstanceAdaptor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        configuration = {}

        return {
            **super(InstanceAdaptor, self).get_config(),
            **configuration
        }
