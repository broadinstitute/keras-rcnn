import keras.backend
import numpy
import tensorflow

import keras_rcnn.backend.tensorflow_backend
import keras_rcnn.backend.common


def test_transpose():
    x = numpy.arange(4).reshape((2, 2))

    target = numpy.transpose(x)

    output = keras_rcnn.backend.transpose(x)

    output = keras.backend.eval(output)

    numpy.testing.assert_array_equal(target, output)

    x = numpy.ones((1, 2, 3))

    target = numpy.transpose(x, (1, 0, 2))

    output = keras_rcnn.backend.transpose(x, [1, 0, 2])

    output = keras.backend.eval(output)

    numpy.testing.assert_array_equal(target, output)


def test_shuffle():
    x = keras.backend.variable(numpy.random.random((10,)))

    keras_rcnn.backend.shuffle(x)


def test_matmul():
    pass


def test_gather_nd():
    pass


def test_argsort():
    pass


def test_scatter_add_tensor():
    ref = keras.backend.ones((4, 5))
    ii = keras.backend.reshape(
        keras.backend.cast(keras.backend.zeros((4,)), 'int32'), (-1, 1))
    jj = keras.backend.reshape(keras.backend.arange(0, 4), (-1, 1))
    indices = keras.backend.concatenate([ii, jj], 1)
    updates = keras.backend.arange(4, dtype=keras.backend.floatx()) * 2
    result = keras_rcnn.backend.scatter_add_tensor(ref, indices, updates)
    result = keras.backend.eval(result)
    expected = numpy.ones((4, 5))
    expected[0, :4] += numpy.arange(4) * 2
    numpy.testing.assert_array_almost_equal(result, expected)


def test_meshgrid():
    pass


def test_unique():
    pass


def test_smooth_l1():
    pass


def test_where():
    pass


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
        numpy.array([[0.1, 0.1, 0.2, 0.2],
                     [0.5, 0.5, 0.8, 0.8]]))

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
