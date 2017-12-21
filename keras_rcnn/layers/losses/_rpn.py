# -*- coding: utf-8 -*-

import keras.backend
import keras.layers

import keras_rcnn.backend


class RPN(keras.layers.Layer):
    def __init__(self, anchors=9, **kwargs):
        self.anchors = anchors

        super(RPN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target_deltas, target_scores, output_deltas, output_scores = inputs

        self.target_deltas = target_deltas
        self.target_scores = target_scores

        self.output_deltas = output_deltas
        self.output_scores = output_scores

        self.output_deltas = keras.backend.reshape(output_deltas, (1, -1, 4))
        self.output_scores = keras.backend.reshape(output_scores, (1, -1))

        self.add_loss(self.classification_loss + self.regression_loss)

        return [output_deltas, output_scores]

    @property
    def classification_loss(self):
        condition = keras.backend.not_equal(self.target_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        target = keras_rcnn.backend.gather_nd(self.target_scores, indices)
        output = keras_rcnn.backend.gather_nd(self.output_scores, indices)

        loss = keras.backend.binary_crossentropy(target, output)

        return keras.backend.mean(loss)

    @property
    def regression_loss(self):
        condition = keras.backend.not_equal(self.output_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(self.output_deltas, indices)
        target = keras_rcnn.backend.gather_nd(self.target_deltas, indices)

        output_scores = keras_rcnn.backend.gather_nd(self.output_scores, indices)

        condition = keras.backend.greater(output_scores, 0)

        x = keras.backend.zeros_like(output_scores) + 1
        y = keras.backend.zeros_like(output_scores)

        p_star_i = keras_rcnn.backend.where(condition, x, y)

        p_star_i = keras.backend.expand_dims(p_star_i, 0)

        output = keras.backend.expand_dims(output, 0)
        target = keras.backend.expand_dims(target, 0)

        a_y = keras_rcnn.backend.smooth_l1(output, target, anchored=True)

        a = p_star_i * a_y

        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.sum(p_star_i + keras.backend.epsilon())

        weight = 10.0

        return weight * (a / b)

    def get_config(self):
        configuration = {
            "anchors": self.anchors
        }

        return {
            **super(RPN, self).get_config(), **configuration
        }
