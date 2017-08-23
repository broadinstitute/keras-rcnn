import keras.backend
import keras.utils
import numpy

import keras_rcnn.layers


class TestBoxRegression:
    def test_call(self):
        classes = 3
        proposal_target = keras_rcnn.layers.BoxRegression()

        proposals = numpy.random.random((1, 100, 4))
        proposals = keras.backend.variable(proposals)

        pred_boxes = numpy.random.choice(range(0, 224), (1, 300, 4 * classes))
        pred_boxes = keras.backend.variable(pred_boxes)

        pred_scores = numpy.random.random((1, 300, classes))
        pred_scores = keras.backend.variable(pred_scores)

        metadata = keras.backend.variable([[224, 224, 1.5]])

        proposal_target.call([proposals, pred_boxes, pred_scores, metadata])

