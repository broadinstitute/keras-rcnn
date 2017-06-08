import numpy
import keras.backend

import keras_rcnn.losses.rcnn


def test_classification():
    y_pred = keras.backend.variable(numpy.random.random((1, 4, 4)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4)))
    assert keras.backend.eval(keras_rcnn.losses.rcnn.classification(y_true, y_pred)).shape == ()


def test_regression():
    n_classes = 9
    rcnn_reg = keras_rcnn.losses.rcnn.regression(n_classes)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4 * n_classes)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 8 * n_classes)))
    expected_loss = numpy.power(0.5, 3)
    loss = keras.backend.eval(rcnn_reg(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)
