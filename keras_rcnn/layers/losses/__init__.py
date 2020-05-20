# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend
from ._mask_rcnn import RCNNMaskLoss


class RCNN(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__(**kwargs)

    def classification_loss(self):
        loss = keras_rcnn.backend.softmax_classification(
            self.target_scores, self.output_scores, anchored=True
        )

        return tensorflow.keras.backend.mean(loss)

    def regression_loss(self):
        output_deltas = self.output_deltas[:, :, 4:]
        target_deltas = self.target_deltas[:, :, 4:]

        # mask out output values where class is different from targetrcnn loss
        # function
        mask = self.target_scores

        labels = tensorflow.keras.backend.repeat_elements(mask, 4, -1)
        labels = labels[:, :, 4:]

        loss = keras_rcnn.backend.smooth_l1(
            output_deltas * labels, target_deltas * labels, anchored=True
        )

        target_scores = self.target_scores[:, :, 1:]

        return tensorflow.keras.backend.sum(loss) / tensorflow.keras.backend.maximum(
            tensorflow.keras.backend.epsilon(),
            tensorflow.keras.backend.sum(target_scores),
        )

    def call(self, inputs, **kwargs):
        target_deltas, target_scores, output_deltas, output_scores = inputs

        self.target_deltas = target_deltas
        self.target_scores = target_scores

        self.output_deltas = output_deltas
        self.output_scores = output_scores

        loss = self.classification_loss() + self.regression_loss()

        weight = 1.0

        loss = weight * loss

        self.add_loss(loss)

        return [output_deltas, output_scores]


class RPN(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RPN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target_deltas, target_scores, output_deltas, output_scores = inputs

        a = self.classification_loss(target_scores, output_scores)

        b = self.regression_loss(target_deltas, target_scores, output_deltas)

        weight = 1.0

        loss = weight * (a + b)

        self.add_loss(loss)

        return [output_deltas, output_scores]

    @staticmethod
    def classification_loss(target_scores, output_scores):
        output_scores = tensorflow.keras.backend.reshape(output_scores, (1, -1))

        condition = tensorflow.keras.backend.not_equal(target_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        indices = tensorflow.keras.backend.expand_dims(indices, 0)

        target = keras_rcnn.backend.gather_nd(target_scores, indices)
        output = keras_rcnn.backend.gather_nd(output_scores, indices)

        loss = tensorflow.keras.backend.binary_crossentropy(target, output)

        return tensorflow.keras.backend.mean(loss)

    @staticmethod
    def regression_loss(target_deltas, target_scores, output_deltas):
        output_deltas = tensorflow.keras.backend.reshape(output_deltas, (1, -1, 4))

        condition = tensorflow.keras.backend.not_equal(target_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output_deltas, indices)
        target = keras_rcnn.backend.gather_nd(target_deltas, indices)

        target_scores = keras_rcnn.backend.gather_nd(target_scores, indices)

        condition = tensorflow.keras.backend.greater(target_scores, 0)

        x = tensorflow.keras.backend.zeros_like(target_scores) + 1
        y = tensorflow.keras.backend.zeros_like(target_scores)

        p_star_i = keras_rcnn.backend.where(condition, x, y)

        p_star_i = tensorflow.keras.backend.expand_dims(p_star_i, 0)

        output = tensorflow.keras.backend.expand_dims(output, 0)
        target = tensorflow.keras.backend.expand_dims(target, 0)

        a_y = keras_rcnn.backend.smooth_l1(output, target, anchored=True)

        a = p_star_i * a_y

        # Divided by anchor overlaps
        weight = 1.0

        loss = weight * (
            tensorflow.keras.backend.sum(a)
            / tensorflow.keras.backend.maximum(
                tensorflow.keras.backend.epsilon(),
                tensorflow.keras.backend.sum(p_star_i),
            )
        )

        return loss

    # COMPUTE OUTPUT SHAPE
