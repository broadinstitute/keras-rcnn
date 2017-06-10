import numpy
import keras.layers
import keras.backend
import keras_rcnn.backend


def test_crop_and_resize():
    image = keras.backend.variable(
        numpy.array([
            numpy.random.random((28, 28, 3)),
            numpy.ones((28, 28, 3))]))
    boxes = keras.backend.variable(
        numpy.array([
            [[0.3, 0.3, 0.8, 0.9],
             [0.5, 0.1, 0.7, 0.5]],
            [[0.1, 0.1, 0.2, 0.2],
             [0.5, 0.5, 0.8, 0.8]]]))
    size = [7, 7]
    slices = keras_rcnn.backend.crop_and_resize(image, boxes, size)
    values = keras.backend.eval(slices)
    assert numpy.allclose(values[-1:], 1.)
    assert values.shape == (2, 2, 7, 7, 3)
