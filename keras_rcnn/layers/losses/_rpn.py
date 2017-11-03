# -*- coding: utf-8 -*-

import keras.backend
import keras.layers

import keras_rcnn.backend


class RPNClassificationLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RPNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs
        output = keras.backend.reshape(output, [1, -1, 2])

        loss = self.compute_loss(output, target)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target):

        condition = keras.backend.not_equal(target, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)
        loss = keras.backend.sparse_categorical_crossentropy(target, output)
        loss = keras.backend.mean(loss)

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class RPNRegressionLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RPNRegressionLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target, labels = inputs

        loss = self.compute_loss(output, target, labels)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target, labels):
        # Robust L1 Loss
        output = keras.backend.reshape(output, [1, -1, 4])

        condition = keras.backend.not_equal(labels, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)
        labels = keras_rcnn.backend.gather_nd(labels, indices)

        p_star_i = keras_rcnn.backend.where(keras.backend.greater(labels, 0), keras.backend.ones_like(labels), keras.backend.zeros_like(labels))

        p_star_i = keras.backend.expand_dims(p_star_i, 0)

        a_y = keras_rcnn.backend.smooth_l1(keras.backend.expand_dims(output, 0), keras.backend.expand_dims(target, 0), anchored=True)

        a = p_star_i * a_y

        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.sum(p_star_i + keras.backend.epsilon())

        loss = 1.0 * (a / b)

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]
