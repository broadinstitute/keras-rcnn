import keras.backend
import numpy
import tensorflow

import keras_rcnn.backend.tensorflow_backend
import keras_rcnn.backend.common


def test_shuffle():
    x = keras.backend.variable(numpy.random.random((10,)))

    keras_rcnn.backend.shuffle(x)


def test_non_max_suppression():
    boxes = numpy.zeros((1764, 4))
    scores = numpy.random.rand(14 * 14, 9).flatten()
    threshold = 0.5
    maximum = 100
    nms = tensorflow.image.non_max_suppression(boxes=boxes,
                                               iou_threshold=threshold,
                                               max_output_size=maximum,
                                               scores=scores)
    assert keras.backend.eval(nms).shape == (maximum,)


def test_crop_and_resize():
    image = keras.backend.variable(numpy.ones((1, 28, 28, 3)))

    boxes = keras.backend.variable(
        numpy.array([[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.8, 0.8]]]))

    size = [7, 7]

    slices = keras_rcnn.backend.crop_and_resize(image, boxes, size)

    assert keras.backend.eval(slices).shape == (2, 7, 7, 3)


def test_squeeze():
    x = [[[0], [1], [2]]]

    x = keras.backend.variable(x)

    assert keras.backend.int_shape(x) == (1, 3, 1)

    y = keras_rcnn.backend.tensorflow_backend.squeeze(x)

    assert keras.backend.int_shape(y) == (3,)

    y = keras_rcnn.backend.tensorflow_backend.squeeze(x, 0)

    assert keras.backend.int_shape(y) == (3, 1)

    y = keras_rcnn.backend.tensorflow_backend.squeeze(x, 2)

    assert keras.backend.int_shape(y) == (1, 3)
