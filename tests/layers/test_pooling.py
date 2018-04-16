import numpy
import keras.backend
import keras_rcnn.layers
import mock


def test_roi():
    image = keras.backend.variable(numpy.random.random((1, 28, 14, 3)))
    boxes = keras.backend.variable(numpy.array([[[1, 2, 3, 4],
                                                 [4, 3, 2, 1]]]))

    metadata = keras.backend.variable([[28, 14, 3]])

    roi_align = keras_rcnn.layers.RegionOfInterest(extent=[7, 7], strides=1)

    slices = roi_align([metadata, image, boxes])

    assert keras.backend.eval(slices).shape == (1, 2, 7, 7, 3)

    with mock.patch("keras_rcnn.backend.crop_and_resize",
                    lambda x, y, z: y):
        boxes = roi_align([metadata, image, boxes])
        numpy.testing.assert_array_almost_equal(
            keras.backend.eval(boxes),
            [[[2. / 28, 1. / 14, 4. / 28, 3. / 14],
              [3. / 28, 4. / 14, 1. / 28, 2. / 14]]])

    a = keras.backend.placeholder(shape=(None, 224, 224, 3))
    b = keras.backend.placeholder(shape=(1, None, 4))
    y = keras_rcnn.layers.RegionOfInterest([7, 7])([metadata, a, b])
    # Should be (None, None, 7, 7, 3)
    assert keras.backend.int_shape(y) == (1, None, 7, 7, 3)
