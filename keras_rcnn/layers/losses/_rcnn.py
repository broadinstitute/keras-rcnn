# -*- coding: utf-8 -*-

import keras.backend
import keras.layers
import tensorflow

import keras_rcnn.backend


class RCNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output_deltas, target_deltas, output_scores, target_scores = inputs

        loss = 0.0

        self.add_loss(loss, inputs)

        return [output_deltas, output_scores]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @staticmethod
    def compute_classification_loss(output, target):
        pass

    @staticmethod
    def compute_regression_loss(output, target):
        pass


class RCNNClassificationLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs

        def no_loss():
            return keras.backend.constant(0.0)
        
        def calculate_loss():
            return self.compute_classification_loss(output, target)

        x = keras.backend.shape(output)[1]
        y = keras.backend.shape(target)[1]

        predicate = keras.backend.not_equal(x, y)

        loss = tensorflow.cond(predicate, no_loss, calculate_loss)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_classification_loss(output, target):
        loss = keras_rcnn.backend.softmax_classification(output, target, anchored=True)

        loss = keras.backend.mean(loss)

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class RCNNRegressionLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNRegressionLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target, labels_target = inputs

        def no_loss():
            return keras.backend.constant(0.0)
        
        def calculate_loss():
            return self.compute_regression_loss(output, target, labels_target)

        x = keras.backend.shape(output)[1]
        y = keras.backend.shape(target)[1]

        predicate = keras.backend.not_equal(x, y)

        loss = tensorflow.cond(predicate, no_loss, calculate_loss)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_regression_loss(output, target_bounding_boxes, target_labels):
        """
        Return the regression loss of Fast R-CNN.
        :return: A loss function for R-CNNs.
        """
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

        # TODO: refactor to use `keras_rcnn.backend.smooth_l1`:
        inside_mul = inside_weights * keras.backend.abs(output - target_bounding_boxes) * labels
        smooth_l1_sign = keras.backend.cast(keras.backend.less(inside_mul, 1.0 / sigma2), keras.backend.floatx())

        smooth_l1_option1 = (inside_mul * inside_mul) * (0.5 * sigma2)
        smooth_l1_option2 = inside_mul - (0.5 / sigma2)

        smooth_l1_result = (smooth_l1_option1 * smooth_l1_sign)
        smooth_l1_result += (smooth_l1_option2 * (1.0 - smooth_l1_sign))

        loss = outside_weights * smooth_l1_result
        epsilon = 1e-4
        b = keras.backend.sum(epsilon + labels)
        loss = tensorflow.reduce_sum(loss) / b

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]
