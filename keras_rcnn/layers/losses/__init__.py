# -*- coding: utf-8 -*-

import keras.backend
import keras.layers

import keras_rcnn.backend


class RCNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__(**kwargs)

    def classification_loss(self):
        loss = keras_rcnn.backend.softmax_classification(self.target_scores, self.output_scores, anchored=True)

        return keras.backend.mean(loss)

    def regression_loss(self):
        output_deltas = self.output_deltas[:, :, 4:]
        target_deltas = self.target_deltas[:, :, 4:]

        # mask out output values where class is different from targetrcnn loss
        # function
        mask = self.target_scores

        labels = keras.backend.repeat_elements(mask, 4, -1)
        labels = labels[:, :, 4:]

        loss = keras_rcnn.backend.smooth_l1(output_deltas * labels, target_deltas * labels, anchored=True)

        target_scores = self.target_scores[:, :, 1:]

        return keras.backend.sum(loss) / keras.backend.maximum(keras.backend.epsilon(), keras.backend.sum(target_scores))

    def call(self, inputs, **kwargs):
        target_deltas, target_scores, output_deltas, output_scores = inputs

        self.target_deltas = target_deltas
        self.target_scores = target_scores

        self.output_deltas = output_deltas
        self.output_scores = output_scores

        loss = self.classification_loss() + self.regression_loss()

        self.add_loss(loss)

        return [output_deltas, output_scores]

    def get_config(self):
        configuration = {}

        return {**super(RCNN, self).get_config(), **configuration}


class RPN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RPN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target_deltas, target_scores, output_deltas, output_scores = inputs

        a = self.classification_loss(target_scores, output_scores)

        b = self.regression_loss(target_deltas, target_scores, output_deltas)

        loss = a + b

        self.add_loss(loss)

        return [output_deltas, output_scores]

    @staticmethod
    def classification_loss(target_scores, output_scores):
        output_scores = keras.backend.reshape(output_scores, (1, -1))

        condition = keras.backend.not_equal(target_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        indices = keras.backend.expand_dims(indices, 0)

        target = keras_rcnn.backend.gather_nd(target_scores, indices)
        output = keras_rcnn.backend.gather_nd(output_scores, indices)

        loss = keras.backend.binary_crossentropy(target, output)

        return keras.backend.mean(loss)

    @staticmethod
    def regression_loss(target_deltas, target_scores, output_deltas):
        output_deltas = keras.backend.reshape(output_deltas, (1, -1, 4))

        condition = keras.backend.not_equal(target_scores, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output_deltas, indices)
        target = keras_rcnn.backend.gather_nd(target_deltas, indices)

        target_scores = keras_rcnn.backend.gather_nd(target_scores, indices)

        condition = keras.backend.greater(target_scores, 0)

        x = keras.backend.zeros_like(target_scores) + 1
        y = keras.backend.zeros_like(target_scores)

        p_star_i = keras_rcnn.backend.where(condition, x, y)

        p_star_i = keras.backend.expand_dims(p_star_i, 0)

        output = keras.backend.expand_dims(output, 0)
        target = keras.backend.expand_dims(target, 0)

        a_y = keras_rcnn.backend.smooth_l1(output, target, anchored=True)

        a = p_star_i * a_y

        # Divided by anchor overlaps
        weight = 10.0

        loss = weight * (keras.backend.sum(a) / keras.backend.maximum(keras.backend.epsilon(), keras.backend.sum(p_star_i)))

        return loss

    def get_config(self):
        configuration = {}

        return {**super(RPN, self).get_config(), **configuration}
