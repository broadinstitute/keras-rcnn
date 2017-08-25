import keras.backend
import numpy

import keras_rcnn.backend
import keras_rcnn.layers
import keras_rcnn.layers.object_detection


def test_rcnn_classification():
    num_classes = 5
    layer = keras_rcnn.layers.losses.RCNNClassificationLoss()
    classes = numpy.random.choice(range(0, num_classes), (91))
    target = numpy.zeros((1, 91, num_classes))
    target[0, numpy.arange(91), classes] = 1
    target = keras.backend.variable(target)
    scores = keras.backend.variable(numpy.random.random((1, 91, num_classes)))

    numpy.testing.assert_array_equal(layer.call([target, scores]), scores)

    assert len(layer.losses) == 1


def test_rcnn_regression():
    num_classes = 5
    layer = keras_rcnn.layers.losses.RCNNRegressionLoss()

    deltas = numpy.zeros((1, 91, 4 * num_classes))
    target = numpy.zeros((1, 91, 4 * num_classes))

    expected_loss = 0

    numpy.testing.assert_array_equal(layer.call([target, deltas]), deltas)

    assert len(layer.losses) == 1

    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)

    deltas = numpy.array([[1, 0, 1, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                          [0, 0, 0, 0, 7, 3, 5, 6, 4, 1, 1, 5],
                          [5, 2, 0, 0, 2, 0, 7, 8, 1, 5, 2, 10],
                          [0, 0, 0, 0, 5, 3, 13, 4, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    deltas = numpy.expand_dims(deltas, 0)
    deltas = keras.backend.variable(deltas)
    target = numpy.array([[0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                          [0, 0, 0, 0, 7, 3, 5, 6, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 10],
                          [0, 0, 0, 0, 5, 3, 13, 4, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 40, 12, 2, .5]])
    target = numpy.expand_dims(target, 0)
    target = keras.backend.variable(target)

    numpy.testing.assert_array_equal(layer.call([target, deltas]), deltas)

    expected_loss = 52.625

    assert keras.backend.eval(layer.losses[1]) == expected_loss
