# -*- coding: utf-8 -*-

import keras.backend
import keras.layers

import keras_rcnn.backend


class RPN(keras.layers.Layer):
    def __init__(self, anchors=9, **kwargs):
        self.anchors = anchors

        super(RPN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output_bounding_boxes, target_bounding_boxes, output_scores, target_scores = inputs

        output_scores = keras.backend.reshape(output_scores, (1, -1))

        classification_loss = self.compute_classification_loss(target_scores, output_scores)

        output_bounding_boxes = keras.backend.reshape(output_bounding_boxes, (1, -1, 4))

        regression_loss = self.compute_regression_loss(target_bounding_boxes, output_bounding_boxes, output_scores)

        loss = classification_loss + regression_loss

        self.add_loss(loss)

        return [output_bounding_boxes, output_scores]

    @staticmethod
    def compute_classification_loss(target, output):
        condition = keras.backend.not_equal(target, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)

        loss = keras.backend.binary_crossentropy(target, output)
        loss = keras.backend.mean(loss)

        return loss

    @staticmethod
    def compute_regression_loss(target, output, output_scores):
        condition = keras.backend.not_equal(output_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)
        output_scores = keras_rcnn.backend.gather_nd(output_scores, indices)

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

        loss = weight * (a / b)

        return loss

    def get_config(self):
        configuration = {
            "anchors": self.anchors
        }

        return {**super(RPN, self).get_config(), **configuration}
