import numpy
import keras.backend
import keras_rcnn.backend


def test_crop_and_resize():
    image = keras.backend.variable(numpy.ones((1, 28, 28, 3)))
    boxes = keras.backend.variable(
        numpy.array([[[0.1, 0.1, 0.2, 0.2],
                      [0.5, 0.5, 0.8, 0.8]]]))
    size = [7, 7]
    slices = keras_rcnn.backend.crop_and_resize(image, boxes, size)
    values = keras.backend.eval(slices)
    assert numpy.allclose(values, 1.)
    assert values.shape == (2, 7, 7, 3)


def test_overlap():
    x = numpy.asarray([
        [0, 10, 0, 10],
        [0, 20, 0, 20],
        [0, 30, 0, 30],
        [0, 40, 0, 40],
        [0, 50, 0, 50],
        [0, 60, 0, 60],
        [0, 70, 0, 70],
        [0, 80, 0, 80],
        [0, 90, 0, 90]
    ])

    y = numpy.asarray([
        [0, 20, 0, 20],
        [0, 40, 0, 40],
        [0, 60, 0, 60],
        [0, 80, 0, 80]
    ])

    overlapping = keras_rcnn.backend.overlap(x, y)

    expected = numpy.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    numpy.testing.assert_array_equal(expected, overlapping)
