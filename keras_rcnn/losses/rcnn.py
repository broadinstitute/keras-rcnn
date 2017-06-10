import keras.backend
import keras.losses


def classification(y_true, y_pred):
    """
    Classification loss of Faster R-CNN.

    :param y_true:
    :param y_pred:

    :return:
    """
    return keras.backend.mean(keras.losses.categorical_crossentropy(y_true, y_pred))


def regression(classes):
    """
    Return the regression loss of Faster R-CNN.

    :param classes: Integer, number of classes.

    :return: A loss function for R-CNNs.
    """
    def f(y_true, y_pred):
        x = y_true[:, :, 4 * classes:] - y_pred
        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = y_true[:, :, :4 * classes]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f
