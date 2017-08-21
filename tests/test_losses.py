import keras.backend
import numpy

import keras_rcnn.backend


def test_smooth_l1_loss():
    y_true = numpy.zeros((1, 4, 4))
    y_pred = numpy.array([[[0, 0, 0, 0], [1, 0, 0, 1], [0, 0.1, 0.5, 0.5], [0, 0, 0, 2]]])

    loss = keras_rcnn.backend.smooth_l1_loss(y_true, y_pred)

    loss = keras.backend.eval(loss)
    assert loss == 2.755
