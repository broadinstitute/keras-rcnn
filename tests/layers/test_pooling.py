import numpy
import keras.backend
import keras_rcnn.layers


def test_roi_align():
    image = keras.backend.variable(numpy.random.random((1, 28, 28, 3)))
    regions = keras.backend.variable(numpy.array([[1, 2, 3, 4],
                                                  [4, 3, 2, 1]]))
    roi_align = keras_rcnn.layers.ROIAlign(size=[7, 7],
                                           regions=2, stride=1)
    slices = roi_align([image, regions])
    assert keras.backend.eval(slices).shape == (2, 7, 7, 3)
