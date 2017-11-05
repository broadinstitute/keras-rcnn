# -*- coding: utf-8 -*-

import keras.backend
import keras.layers
import tensorflow

import keras_rcnn.backend


class RCNNClassificationLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs

        def no_loss():
            return keras.backend.constant(0.0)
        
        def calculate_loss():
            return self.compute_classification_loss(output, target)
        
        loss = tensorflow.cond(keras.backend.not_equal(keras.backend.shape(output)[1], keras.backend.shape(target)[1]), no_loss, calculate_loss)

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
        
        loss = tensorflow.cond(keras.backend.not_equal(keras.backend.shape(output)[1], keras.backend.shape(target)[1]), no_loss, calculate_loss)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_regression_loss(output, target, labels_target):
        """
        Return the regression loss of Faster R-CNN.
        :return: A loss function for R-CNNs.
        """
        inside_weights = 1.0
        outside_weights = 1.0
        sigma = 1.0
        sigma2 = keras.backend.square(sigma)

        # only consider positive classes
        output = output[:, :, 4:]
        target = target[:, :, 4:]
        labels_target = labels_target[:, :, 1:]

        # mask out output values where class is different from targetrcnn loss
        # function
        a = keras_rcnn.backend.where(keras.backend.equal(labels_target, 1))
        a = keras.backend.cast(a, 'int32')

        indices_r = a[:, :2]
        indices_c = a[:, 2:]
        indices_0 = keras.backend.concatenate([indices_r, indices_c * 4], 1)
        indices_1 = keras.backend.concatenate([indices_r, indices_c * 4 + 1], 1)
        indices_2 = keras.backend.concatenate([indices_r, indices_c * 4 + 2], 1)
        indices_3 = keras.backend.concatenate([indices_r, indices_c * 4 + 3], 1)
        indices = keras.backend.concatenate([indices_0,
                                            indices_1,
                                            indices_2,
                                            indices_3], 0)
        updates = keras.backend.ones_like(indices, dtype=keras.backend.floatx())
        labels = keras_rcnn.backend.scatter_add_tensor(keras.backend.zeros_like(output, dtype='float32'), indices, updates[:, 0])

        inside_mul = inside_weights * keras.backend.abs(output - target) * labels
        smooth_l1_sign = keras.backend.cast(keras.backend.less(inside_mul, 1.0 / sigma2), keras.backend.floatx())

        smooth_l1_option1 = (inside_mul * inside_mul) * (0.5 * sigma2)
        smooth_l1_option2 = inside_mul - (0.5 / sigma2)

        smooth_l1_result = (smooth_l1_option1 * smooth_l1_sign)
        smooth_l1_result += (smooth_l1_option2 * (1.0 - smooth_l1_sign))

        loss = outside_weights * smooth_l1_result
        epsilon = 1e-4
        b = keras.backend.sum(epsilon + labels_target)
        loss = tensorflow.reduce_sum(loss) / b

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]
