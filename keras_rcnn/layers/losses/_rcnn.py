import keras.layers
import tensorflow

import keras_rcnn.backend


class RCNNClassificationLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target, output = inputs

        loss = self.compute_loss(target, output)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(target, output):
        """
        Classification loss of Faster R-CNN.

        :param target:
        :param output:

        :return:
        """
        return keras.backend.mean(keras.losses.categorical_crossentropy(target, output))

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class RCNNRegressionLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNRegressionLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target, output = inputs

        loss = self.compute_loss(target, output)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(target, output):
        """
        Return the regression loss of Faster R-CNN.

        :return: A loss function for R-CNNs.
        """
        inside_weights = 1.0
        outside_weights = 1.0
        sigma = 1.0
        sigma2 = sigma * sigma
        inside_mul = tensorflow.multiply(inside_weights, tensorflow.subtract(output, target))
        smooth_l1_sign = tensorflow.cast(tensorflow.less(tensorflow.abs(inside_mul), 1.0 / sigma2), tensorflow.float32)
        smooth_l1_option1 = tensorflow.multiply(tensorflow.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tensorflow.subtract(tensorflow.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tensorflow.add(tensorflow.multiply(smooth_l1_option1, smooth_l1_sign), tensorflow.multiply(smooth_l1_option2, tensorflow.abs(tensorflow.subtract(smooth_l1_sign, 1.0))))
        loss = tensorflow.multiply(outside_weights, smooth_l1_result)
        loss = tensorflow.reduce_mean(tensorflow.reduce_sum(loss, reduction_indices=[1]))

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[1]
