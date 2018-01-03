# -*- coding: utf-8 -*-

import keras.backend
import keras.layers
import tensorflow

import keras_rcnn.backend

from ._mask_rcnn import RCNNMaskLoss


class RCNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__(**kwargs)

    @property
    def classification_loss(self):
        epsilon = keras.backend.epsilon()

        weights = keras.backend.sum(self.target_scores, axis=1)

        weights = keras.backend.sum(self.target_scores) / keras.backend.maximum(weights, epsilon)

        weights = 1.0 / (1 + keras.backend.exp(-weights))

        weights = keras.backend.sum(self.target_scores * weights, axis=-1)

        loss = keras_rcnn.backend.softmax_classification(self.output_scores, self.target_scores, anchored=True, weights=weights)

        return keras.backend.mean(loss)

    @property
    def regression_loss(self):
        output_deltas = self.output_deltas[:, :, 4:]
        target_deltas = self.target_deltas[:, :, 4:]

        target_scores = self.target_scores[:, :, 1:]

        weights = keras.backend.sum(target_scores, axis=1)

        weights = keras.backend.sum(target_scores) / keras.backend.maximum(weights, keras.backend.epsilon())

        weights = 1.0 / (1 + keras.backend.exp(-weights))

        # mask out output values where class is different from targetrcnn loss
        # function
        a = keras_rcnn.backend.where(keras.backend.equal(target_scores, 1))

        a = keras.backend.cast(a, 'int32')

        rr = a[:, :2]
        cc = a[:, 2:]

        indices = [
            keras.backend.concatenate([rr, cc * 4 + 0], 1),
            keras.backend.concatenate([rr, cc * 4 + 1], 1),
            keras.backend.concatenate([rr, cc * 4 + 2], 1),
            keras.backend.concatenate([rr, cc * 4 + 3], 1)
        ]

        indices = keras.backend.concatenate(indices, 0)

        weights = keras.backend.sum(target_scores * weights, axis=-1)
        weights = keras_rcnn.backend.gather_nd(weights, rr)

        weights = keras.backend.reshape(weights, (-1,))

        updates = keras.backend.tile(weights, (4,))

        labels = keras_rcnn.backend.scatter_add_tensor(keras.backend.zeros_like(output_deltas, dtype='float32'), indices, updates)

        loss = keras_rcnn.backend.smooth_l1(output_deltas * labels, target_deltas * labels, anchored=True)

        return keras.backend.sum(loss) / keras.backend.maximum(keras.backend.epsilon(), keras.backend.sum(target_scores))

    def call(self, inputs, **kwargs):
        target_deltas, target_scores, output_deltas, output_scores = inputs

        self.target_deltas = target_deltas
        self.target_scores = target_scores

        self.output_deltas = output_deltas
        self.output_scores = output_scores

        def backward():
            return self.classification_loss + self.regression_loss

        def forward():
            return keras.backend.constant(0.0)

        target_deltas_x = keras.backend.shape(self.target_deltas)[1]
        target_scores_x = keras.backend.shape(self.target_scores)[1]

        output_deltas_y = keras.backend.shape(self.output_deltas)[1]
        output_scores_y = keras.backend.shape(self.output_scores)[1]

        a = keras.backend.not_equal(target_deltas_x, output_deltas_y)
        b = keras.backend.not_equal(target_scores_x, output_scores_y)

        loss = tensorflow.cond(a, forward, backward)

        self.add_loss(loss)

        return [output_deltas, output_scores]

    def get_config(self):
        configuration = {}

        return {**super(RCNN, self).get_config(), **configuration}


class RPN(keras.layers.Layer):
    def __init__(self, anchors=9, **kwargs):
        self.anchors = anchors

        super(RPN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output_deltas, target_deltas, output_scores, target_scores = inputs

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

        indices = keras.backend.expand_dims(indices, 0)

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

        # Divided by anchor overlaps
        weight = 10.0

        return weight * (keras.backend.sum(a) / keras.backend.maximum(keras.backend.epsilon(), keras.backend.sum(p_star_i)))

    def get_config(self):
        configuration = {
            "anchors": self.anchors
        }

        return {**super(RPN, self).get_config(), **configuration}
