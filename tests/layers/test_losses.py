import keras.backend
import keras_rcnn.layers
import numpy


def test_call_classification():
    anchors = 9
    layer = keras_rcnn.layers.Classification(anchors=anchors)

    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 2 * anchors)))
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, anchors)))

    numpy.testing.assert_array_equal(layer.call([y_true, y_pred]), y_pred)

    assert len(layer.losses) == 1

    expected_loss = -numpy.log(0.5)
    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)


def test_call_regression():
    anchors = 9
    layer = keras_rcnn.layers.Regression(anchors=anchors)

    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 8 * anchors)))
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, 4 * anchors)))

    numpy.testing.assert_array_equal(layer.call([y_true, y_pred]), y_pred)

    assert len(layer.losses) == 1

    expected_loss = numpy.power(0.5, 3)
    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)
