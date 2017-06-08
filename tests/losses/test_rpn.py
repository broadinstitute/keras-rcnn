import keras.backend
import keras.utils
import numpy

import keras_rcnn.losses.rpn


def test_classification():
    n_anchors = 9
    rpn_cls = keras_rcnn.losses.rpn._classification(n_anchors)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, n_anchors)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 2 * n_anchors)))
    expected_loss = - numpy.log(0.5)
    loss = keras.backend.eval(rpn_cls(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)


def test_regression():
    n_anchors = 9
    rpn_reg = keras_rcnn.losses.rpn._regression(n_anchors)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, 4 * n_anchors)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 8 * n_anchors)))
    expected_loss = numpy.power(0.5, 3)
    loss = keras.backend.eval(rpn_reg(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)


# def test_proposal():
#     anchors = 9
#
#     y_true_classification = numpy.random.choice(range(0, 20), 100)
#     y_true_classification = keras.utils.to_categorical(y_true_classification)
#     y_true_classification = numpy.expand_dims(y_true_classification, 0)
#
#     y_true_regression = numpy.random.choice(range(0, 224), 400)
#     y_true_regression = y_true_regression.reshape((-1, 4))
#     y_true_regression = numpy.expand_dims(y_true_regression, 0)
#
#     y_true = numpy.concatenate([y_true_classification, y_true_regression], -1)
#
#     y_pred_classification = numpy.random.random((1, 7, 7, 9))
#     y_pred_regression = numpy.random.random((1, 7, 7, 36))
#
#     y_pred = numpy.concatenate([y_pred_classification, y_pred_regression], -1)
#
#     loss = keras_rcnn.losses.rpn.proposal(anchors)(y_true, y_pred)
#
#     # assert loss == 0.0
