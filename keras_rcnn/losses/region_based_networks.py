import keras.backend as K

from keras.losses import categorical_crossentropy


def rcnn_classification(y_true, y_pred):
    """Classification loss of Faster R-CNN."""
    return K.mean(categorical_crossentropy(y_true, y_pred))


def rcnn_regression(classes):
    """Return the regression loss of Faster R-CNN.

    # Arguments
        classes: Integer, number of classes.
    # Returns
        A loss function for R-CNNs.
    """
    def f(y_true, y_pred):
        x = y_true[:, :, 4 * classes:] - y_pred
        mask = K.less_equal(K.abs(x), 1.0)
        mask = K.cast(mask, K.floatx())

        a_x = y_true[:, :, :4 * classes]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (K.abs(x) - 0.5)

        a = a_x * a_y
        a = K.sum(a)

        # Divided by anchor overlaps
        b = K.epsilon() + a_x
        b = K.sum(b)

        return 1.0 * (a / b)

    return f
