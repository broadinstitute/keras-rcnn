import keras.layers


class Classification(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        self.is_placeholder = True

        super(Classification, self).__init__(**kwargs)

    def _loss(self, y_true, y_pred):
        # Binary classification loss
        x, y = y_pred[:, :, :, :], y_true[:, :, :, self.anchors:]

        a = y_true[:, :, :, :self.anchors] * keras.backend.binary_crossentropy(x, y)
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + y_true[:, :, :, :self.anchors]
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    def call(self, inputs):
        y_true, y_pred = inputs

        loss = self._loss(y_true, y_pred)

        self.add_loss(loss, inputs=inputs)

        return y_pred



class Regression(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        self.is_placeholder = True

        super(Regression, self).__init__(**kwargs)

    def _loss(self, y_true, y_pred):
        # Robust L1 Loss
        x = y_true[:, :, :, 4 * self.anchors:] - y_pred

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = y_true[:, :, :, :4 * self.anchors]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    def call(self, inputs):
        y_true, y_pred = inputs

        loss = self._loss(y_true, y_pred)

        self.add_loss(loss, inputs=inputs)

        return y_pred
