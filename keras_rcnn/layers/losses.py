import keras.layers

import keras_rcnn.backend


class ClassificationLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(ClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target = inputs

        loss = self.compute_loss(output, target)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target):
        output = keras.backend.reshape(output, [-1, 2])

        condition = keras.backend.not_equal(target, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)

        loss = keras.backend.sparse_categorical_crossentropy(output, target)
        loss = keras.backend.mean(loss)

        return loss

    def compute_output_shape(self, input_shape):
        return None, None, None, self.anchors * 2


class RegressionLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RegressionLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, target, labels = inputs

        loss = self.compute_loss(output, target, labels)

        self.add_loss(loss, inputs)

        return output

    @staticmethod
    def compute_loss(output, target, labels):
        # Robust L1 Loss
        output = keras.backend.reshape(output, [-1, 4])

        condition = keras.backend.not_equal(labels, -1)

        indices = keras_rcnn.backend.where(condition)

        output = keras_rcnn.backend.gather_nd(output, indices)
        target = keras_rcnn.backend.gather_nd(target, indices)
        labels = keras_rcnn.backend.gather_nd(labels, indices)

        x = target - output

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = keras_rcnn.backend.where(keras.backend.not_equal(labels, 0), keras.backend.ones_like(labels), keras.backend.zeros_like(labels))
        a_x = keras.backend.cast(a_x, keras.backend.floatx())

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = keras_rcnn.backend.matmul(keras.backend.expand_dims(a_x, 0), a_y)
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        loss = 1.0 * (a / b)

        return loss

    def compute_output_shape(self, input_shape):
        return None, None, None, self.anchors * 4
