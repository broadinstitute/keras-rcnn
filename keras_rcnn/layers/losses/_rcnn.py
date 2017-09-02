import keras.backend
import keras.layers
import tensorflow

import keras_rcnn.backend


class RCNNClassificationLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RCNNClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        output, target = inputs

        loss = keras.backend.in_train_phase(
            lambda: self.compute_loss(output, target),
            keras.backend.variable(0),
            training=training
        )

        self.add_loss(loss, inputs)

        return loss

    @staticmethod
    def compute_loss(output, target):
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

    def call(self, inputs, training=None, **kwargs):
        output, target, labels_target = inputs

        loss = keras.backend.in_train_phase(
            lambda: self.compute_loss(output, target, labels_target),
            keras.backend.variable(0),
            training=training
        )

        self.add_loss(loss, inputs)

        return loss

    @staticmethod
    def compute_loss(output, target, labels_target):
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

        # mask out output values where class is different from targetrcnn loss function
        a = keras.backend.cast(keras_rcnn.backend.where(keras.backend.equal(labels_target, 1)), 'int32')
        indices_r = a[:, :2]
        indices_c = a[:, 2:]
        indices_0 = keras.backend.concatenate([indices_r, indices_c * 4], 1)
        indices_1 = keras.backend.concatenate([indices_r, indices_c * 4 + 1], 1)
        indices_2 = keras.backend.concatenate([indices_r, indices_c * 4 + 2], 1)
        indices_3 = keras.backend.concatenate([indices_r, indices_c * 4 + 3], 1)
        indices = keras.backend.concatenate([indices_0, indices_1, indices_2, indices_3], 0)
        updates = keras.backend.ones_like(indices, dtype=keras.backend.floatx())
        labels = keras_rcnn.backend.scatter_add_tensor(keras.backend.zeros_like(output), indices, updates[:, 0])
        output = output * labels

        inside_mul = inside_weights * keras.backend.abs(output - target)
        smooth_l1_sign = keras.backend.cast(
            keras.backend.less(inside_mul, 1.0 / sigma2),
            keras.backend.floatx())

        smooth_l1_option1 = (inside_mul * inside_mul) * (0.5 * sigma2)
        smooth_l1_option2 = inside_mul - (0.5 / sigma2)

        smooth_l1_result = (smooth_l1_option1 * smooth_l1_sign)
        smooth_l1_result += (smooth_l1_option2 * (1.0 - smooth_l1_sign))

        loss = outside_weights * smooth_l1_result
        epsilon = 1e-4
        b = keras.backend.sum(epsilon + labels)
        loss = tensorflow.reduce_sum(loss)/b

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]

