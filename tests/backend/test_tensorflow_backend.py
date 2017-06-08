import numpy
import keras.backend
import keras_rcnn.backend


def test_crop_and_resize():
    image = keras.backend.variable(numpy.random.random((1, 28, 28, 3)))
    regions = keras.backend.variable(numpy.array([[3, 3, 8, 10],
                                                  [5, 1, 12, 12]]))
    size = [7, 7]
    slices = keras_rcnn.backend.crop_and_resize(image, regions, size)
    assert keras.backend.eval(slices).shape == (2, 7, 7, 3)
