import numpy
import keras.layers
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


def test_propose():
    rpn_scores = keras.backend.variable(numpy.random.random((1, 7, 7, 9)))
    rpn_boxes = keras.backend.variable(numpy.random.random((1, 7, 7, 36)))
    proposals = keras_rcnn.backend.propose(rpn_boxes, rpn_scores, 300)
    assert keras.backend.eval(proposals).shape[0] == 1
    assert keras.backend.eval(proposals).shape[-1] == 4

    proposals = keras_rcnn.backend.propose(rpn_boxes, rpn_scores, 10)
    assert keras.backend.eval(proposals).shape == (1, 10, 4)
