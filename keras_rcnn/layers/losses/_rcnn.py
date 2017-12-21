# -*- coding: utf-8 -*-

import keras.backend
import keras.layers
import tensorflow

import keras_rcnn.backend


class RCNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.target_deltas = None
        self.target_scores = None

        self.output_deltas = None
        self.output_scores = None

        super(RCNN, self).__init__(**kwargs)

    @property
    def classification_loss(self):
        loss = keras_rcnn.backend.softmax_classification(self.target_scores, self.output_scores, anchored=True)

        return keras.backend.mean(loss)

    @property
    def regression_loss(self):
        # NOTE:
        #   - The shape of `output` is `(batches, proposals, 4 × classes)`
        #   - The first four coordinates are reserved for the background class
        target = self.target_deltas[:, :, 4:]
        output = self.output_deltas[:, :, 4:]

        # NOTE:
        #   - The shape of `target_labels` is `(batches, proposals, classes)`
        #   - The first label is reserved for the background class
        target_labels = self.target_scores[:, :, 1:]

        # Since `target_labels` is a one-hot encoded vector we need to find
        # where `target_labels` equals `1` to ensure we’re not including
        # background labels.
        condition = keras.backend.equal(target_labels, 1)

        target_label_indices = keras_rcnn.backend.where(condition)

        # TODO: Do we need this cast?
        target_label_indices = keras.backend.cast(target_label_indices, "int32")

        # For each proposal, multiply the class label by four so that each
        # label has four corresponding bounding box coordinates.
        rr = target_label_indices[:, :2]
        cc = target_label_indices[:, 2:]

        indices = [
            keras.backend.concatenate([rr, cc * 4 + 0], 1),
            keras.backend.concatenate([rr, cc * 4 + 1], 1),
            keras.backend.concatenate([rr, cc * 4 + 2], 1),
            keras.backend.concatenate([rr, cc * 4 + 3], 1)
        ]

        indices = keras.backend.concatenate(indices, 0)

        target_labels = keras.backend.ones_like(indices, dtype=keras.backend.floatx())

        output_labels = keras.backend.zeros_like(output, dtype=keras.backend.floatx())

        output_labels = keras_rcnn.backend.scatter_add_tensor(output_labels, indices, target_labels[:, 0])

        target *= output_labels
        output *= output_labels

        loss = keras_rcnn.backend.smooth_l1(output, target, anchored=True)

        loss = keras.backend.sum(loss) / keras.backend.maximum(keras.backend.sum(output_labels // 4), keras.backend.epsilon())

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

        predicate = tensorflow.logical_and(a, b)

        loss = tensorflow.cond(predicate, forward, backward)

        self.add_loss(loss, inputs)

        return [output_deltas, output_scores]
