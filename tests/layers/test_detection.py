import keras.backend
import keras.utils
import numpy

import keras_rcnn.layers


class TestDetection:
    def test_call(self):
        classes = 3
        proposal_target = keras_rcnn.layers.Detection()

        pred_boxes = numpy.random.random((1, 100, 4 * classes))
        pred_boxes = keras.backend.variable(pred_boxes)

        proposals = numpy.random.choice(range(0, 224), (1, 100, 4))
        proposals = keras.backend.variable(proposals)

        pred_scores = numpy.random.random((1, 100, classes))
        pred_scores = keras.backend.variable(pred_scores)

        metadata = keras.backend.variable([[224, 224, 1.5]])

        boxes, classes = proposal_target.call([proposals, pred_boxes, pred_scores, metadata])

        assert keras.backend.eval(boxes).shape == keras.backend.eval(pred_boxes).shape

        assert keras.backend.eval(classes).shape == keras.backend.eval(boxes).shape

