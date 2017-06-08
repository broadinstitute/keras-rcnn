import keras.backend
import numpy

import keras_rcnn.losses.rpn


def test_classification():
    n_anchors = 9
    rpn_cls = keras_rcnn.losses.rpn.classification(n_anchors)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, n_anchors)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 2 * n_anchors)))
    expected_loss = - numpy.log(0.5)
    loss = keras.backend.eval(rpn_cls(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)


def test_regression():
    n_anchors = 9
    rpn_reg = keras_rcnn.losses.rpn.regression(n_anchors)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, 4 * n_anchors)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 8 * n_anchors)))
    expected_loss = numpy.power(0.5, 3)
    loss = keras.backend.eval(rpn_reg(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)
