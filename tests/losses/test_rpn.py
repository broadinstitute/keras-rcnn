import keras.backend
import keras.utils
import numpy
import keras_rcnn.losses.rpn
import tensorflow


def test_separate_pred():
    anchors = 9
    features = (14, 14)

    y_pred_classification = numpy.zeros((1, features[0], features[1], anchors))
    y_pred_regression = numpy.zeros((1, features[0], features[1], anchors * 4))

    y_pred = numpy.concatenate([y_pred_regression, y_pred_classification], -1)
    y_pred = keras.backend.variable(y_pred)
    assert keras.backend.eval(
        keras_rcnn.losses.rpn.separate_pred(y_pred)[0]).shape == (
           1, features[0], features[1], 4 * anchors)
    assert keras.backend.eval(
        keras_rcnn.losses.rpn.separate_pred(y_pred)[1]).shape == (
           1, features[0], features[1], anchors)


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


def test_encode():
    anchors = 9
    features = (14, 14)
    image_shape = (224, 224)
    samples = 91

    y_true = numpy.random.choice(range(0, image_shape[0]), 4 * samples)
    y_true = y_true.reshape((-1, 4))
    y_true = numpy.expand_dims(y_true, 0)

    y_true = tensorflow.convert_to_tensor(y_true, tensorflow.float32)

    bbox_labels, bbox_reg_targets, inds_inside = keras_rcnn.losses.rpn.encode(
        features, image_shape, y_true)

    assert keras.backend.eval(bbox_labels).shape == (84,)

    assert keras.backend.eval(bbox_reg_targets).shape == (84, 4)

    assert keras.backend.eval(inds_inside).shape == (84,)


def test_proposal():
    anchors = 9
    features = (14, 14)
    image_shape = (224, 224)
    stride = 16

    y_pred_classification = numpy.zeros((1, features[0], features[1], anchors))

    y_pred_regression = numpy.zeros((1, features[0], features[1], anchors * 4))

    y_pred = numpy.concatenate([y_pred_regression, y_pred_classification], -1)

    y_true = numpy.array([
        [1, 1, 100, 100],
        [0, 0, 40, 50],
        [50, 99, 100, 203],
        [111, 5, 131, 34],
        [4, 60, 30, 90]])

    y_true = numpy.expand_dims(y_true, 0)

    y_true = keras.backend.variable(y_true)
    y_pred = keras.backend.variable(y_pred)

    loss1, loss2 = keras_rcnn.losses.rpn.proposal(anchors, image_shape, stride)(y_true, y_pred)
    loss = loss1 + loss2
    loss_value = keras.backend.eval(loss)

    assert numpy.around(loss_value - 16.2735, 3) == 0
    # assert keras.backend.int_shape(loss) == (1, image_shape[0], image_shape[1], 5 * anchors)
