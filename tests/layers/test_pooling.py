import numpy
import keras.backend
import keras_rcnn.layers
from unittest.mock import patch


def test_roi():
    image = keras.backend.variable(numpy.random.random((1, 28, 14, 3)))
    regions = keras.backend.variable(numpy.array([[1, 2, 3, 4],
                                                  [4, 3, 2, 1]]))
    roi_align = keras_rcnn.layers.ROI(size=[7, 7],
                                      regions=2, stride=1)
    slices = roi_align([image, regions])
    assert keras.backend.eval(slices).shape == (2, 7, 7, 3)

    with patch("keras_rcnn.backend.crop_and_resize", lambda x, y, z: y):
        boxes = roi_align([image, regions])
        assert numpy.allclose(
            keras.backend.eval(boxes),
            [[2. / 28,  1. / 14,  6. / 28,  4. / 14, ],
             [3. / 28,  4. / 14,  4. / 28,  6. / 14, ]])
