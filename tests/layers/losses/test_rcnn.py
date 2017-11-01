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

    numpy.testing.assert_array_equal(layer.call([scores, target]), scores)

    assert len(layer.losses) == 1


def test_rcnn_regression():
    num_classes = 5
    layer = keras_rcnn.layers.losses.RCNNRegressionLoss()

    deltas = 1.0 * numpy.zeros((1, 91, 4 * num_classes))
    target = 1.0 * numpy.zeros((1, 91, 4 * num_classes))

    classes = numpy.random.choice(range(0, num_classes), (91))
    labels_target = numpy.zeros((1, 91, num_classes))
    labels_target[0, numpy.arange(91), classes] = 1
    labels_target = keras.backend.variable(labels_target)

    expected_loss = 0

    numpy.testing.assert_array_equal(layer.call([deltas, target, labels_target]), deltas)

    assert len(layer.losses) == 1

    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)

    deltas = numpy.array([[1, 0, 1, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                          [0, 0, 0, 0, 7, 3, 5, 6, 4, 1, 1, 5],
                          [5, 2, 0, 0, 2, 0, 7, 8, 1, 5, 2, 10],
                          [0, 0, 0, 0, 5, 3, 13, 4, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    deltas = 1.0 * numpy.expand_dims(deltas, 0)
    deltas = keras.backend.variable(deltas)
    target = 1.0 * numpy.array([[0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                          [0, 0, 0, 0, 7, 3, 5, 6, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 10],
                          [0, 0, 0, 0, 5, 3, 13, 4, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 40, 12, 2, .5]])
    target = numpy.expand_dims(target, 0)
    target = keras.backend.variable(target)

    labels_target = numpy.array([[1, 0, 0],
                                 [0, 0, 1],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    labels_target = numpy.expand_dims(labels_target, 0)
    labels_target = keras.backend.variable(labels_target)

    output = layer.call([deltas, target, labels_target])
    numpy.testing.assert_array_equal(output, deltas)

    # expected_loss = 52.625
    assert len(layer.losses) == 2
    keras.backend.eval(layer.losses[1])
    # assert numpy.isclose(keras.backend.eval(layer.losses[1]), expected_loss)
