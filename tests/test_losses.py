import keras.backend
import numpy

import keras_rcnn.backend


def test_smooth_l1_loss():
    y_true = numpy.random.random((1, 10, 4))
    y_pred = numpy.random.random((1, 10, 4))

    loss = keras_rcnn.backend.smooth_l1_loss(y_true, y_pred)

    loss = keras.backend.eval(loss)

    import IPython
    IPython.embed()
