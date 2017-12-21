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
        loss = keras_rcnn.backend.softmax_classification(self.target_scores, self.output_scores, anchored=True)

        return keras.backend.mean(loss)

    @property
    def regression_loss(self):
        output = self.output_deltas
        target_bounding_boxes = self.target_deltas
        target_labels = self.target_scores

        # TODO: remove unneeded constants
        inside_weights = 1.0
        outside_weights = 1.0

        sigma = 1.0
        sigma2 = keras.backend.square(sigma)

        # TODO: are we actually removing the negative boxes?
        # only consider positive classes
        output = output[:, :, 4:]
        target_bounding_boxes = target_bounding_boxes[:, :, 4:]

        target_labels = target_labels[:, :, 1:]

        # mask out output values where class is different from targetrcnn loss
        # function
        a = keras_rcnn.backend.where(keras.backend.equal(target_labels, 1))
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

        updates = keras.backend.ones_like(indices, dtype=keras.backend.floatx())

        labels = keras_rcnn.backend.scatter_add_tensor(keras.backend.zeros_like(output, dtype='float32'), indices, updates[:, 0])

        smooth_l1_result = keras_rcnn.backend.smooth_l1(output * labels, target_bounding_boxes * labels, anchored=True)

        loss = outside_weights * smooth_l1_result
        epsilon = 1e-4
        b = keras.backend.sum(epsilon + target_labels)
        loss = tensorflow.reduce_sum(loss) / b

        return loss

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
