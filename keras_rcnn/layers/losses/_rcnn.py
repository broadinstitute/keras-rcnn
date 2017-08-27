import keras.layers
import tensorflow

import keras_rcnn.backend
import keras.backend


class RCNNClassificationLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs

        loss = self.compute_loss(output, target)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target):
        import IPython
        IPython.embed()

        loss = keras_rcnn.backend.softmax_classification(
            output, target, anchored=True
        )

        loss = keras.backend.mean(loss)

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class RCNNRegressionLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNRegressionLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs

        loss = self.compute_loss(output, target)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target):
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

        # mask out output values where class is different from target
        output = keras.backend.cast(output, keras.backend.floatx())
        output = keras_rcnn.backend.where(
            keras.backend.greater(target, 0),
            output,
            keras.backend.zeros_like(target, dtype=keras.backend.floatx())
        )

        inside_mul = inside_weights * (output - target)
        smooth_l1_sign = keras.backend.cast(
            keras.backend.less(keras.backend.abs(inside_mul), 1.0 / sigma2),
            keras.backend.floatx())

        smooth_l1_option1 = (inside_mul * inside_mul) * (0.5 * sigma2)
        smooth_l1_option2 = keras.backend.abs(inside_mul) - (0.5 / sigma2)

        smooth_l1_result = (smooth_l1_option1 * smooth_l1_sign)
        smooth_l1_result += (smooth_l1_option2 * (1.0 - smooth_l1_sign))

        loss = outside_weights * smooth_l1_result

        loss = tensorflow.reduce_sum(loss)

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]
