# -*- coding: utf-8 -*-

import keras.backend
import keras.layers

import keras_rcnn.backend


class RPN(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RPN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output_bounding_boxes, target_bounding_boxes, output_scores, target_scores = inputs

        output_scores = keras.backend.reshape(output_scores, (1, -1))

        classification_loss = self.compute_classification_loss(output_scores, target_scores)

        output_bounding_boxes = keras.backend.reshape(output_bounding_boxes, (1, -1, 4))

        regression_loss = self.compute_regression_loss(output_bounding_boxes, target_bounding_boxes, output_scores)

        loss = classification_loss + regression_loss

        self.add_loss(loss)

        return [output_bounding_boxes, output_scores]

    @staticmethod
    def compute_classification_loss(output, target):
        condition = keras.backend.not_equal(target, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)

        loss = keras.backend.binary_crossentropy(target, output)
        loss = keras.backend.mean(loss)

        return loss

    @staticmethod
    def compute_regression_loss(output, target, labels):
        condition = keras.backend.not_equal(labels, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)
        labels = keras_rcnn.backend.gather_nd(labels, indices)

        condition = keras.backend.greater(labels, 0)

        x = keras.backend.zeros_like(labels) + 1
        y = keras.backend.zeros_like(labels)

        p_star_i = keras_rcnn.backend.where(condition, x, y)

        p_star_i = keras.backend.expand_dims(p_star_i, 0)

        output = keras.backend.expand_dims(output, 0)
        target = keras.backend.expand_dims(target, 0)

        a_y = keras_rcnn.backend.smooth_l1(output, target, anchored=True)

        a = p_star_i * a_y

        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.sum(p_star_i + keras.backend.epsilon())

        loss = 1.0 * (a / b)

        return loss

    def get_config(self):
        configuration = {
            "anchors": self.anchors
        }

        return {**super(RPN, self).get_config(), **configuration}


class RPNClassificationLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RPNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs

        output = keras.backend.reshape(output, (1, -1))

        loss = self.compute_loss(output, target)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target):
        condition = keras.backend.not_equal(target, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)
        loss = keras.backend.binary_crossentropy(target, output)
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

        output = keras.backend.reshape(output, (1, -1, 4))

        loss = self.compute_loss(output, target, labels)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target, labels):
        # Robust L1 Loss
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
