import keras_rcnn.preprocessing._object_detection
import numpy


def test_scale_shape():
    min_size = 200
    max_size = 300
    shape    = (600, 1000, 3)

    shape, scale = keras_rcnn.preprocessing._object_detection.scale_shape(shape, min_size, max_size)

    expected = (180, 300, 3)
    numpy.testing.assert_equal(shape, expected)

    expected = 0.3
    assert numpy.isclose(scale, expected)
