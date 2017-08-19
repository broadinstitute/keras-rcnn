import keras.backend
import numpy

import keras_rcnn.backend
import keras_rcnn.layers


class TestObjectProposal:
    def test_call(self):
        metadata = keras.backend.variable([[224, 224, 1.5]])

        deltas = numpy.random.random((1, 14, 14, 9 * 4))
        scores = numpy.random.random((1, 14, 14, 9 * 2))

        deltas = keras.backend.variable(deltas)
        scores = keras.backend.variable(scores)

        object_proposal = keras_rcnn.layers.ObjectProposal()

        object_proposal.call([metadata, deltas, scores])


def test_bbox_transform_inv():
    anchors = 9
    features = (14, 14)
    shifted = keras_rcnn.backend.shift(features, 16)
    boxes = numpy.zeros((features[0] * features[1] * anchors, 4))
    boxes = keras.backend.variable(boxes)
    pred_boxes = keras_rcnn.layers.object_detection._object_proposal.bbox_transform_inv(shifted, boxes)
    assert keras.backend.eval(pred_boxes).shape == (1764, 4)


def test_filter_boxes():
    proposals = numpy.array(
        [[0, 2, 3, 10],
         [-1, -5, 4, 8],
         [0, 0, 1, 1]]
    )

    minimum = 3

    results = keras_rcnn.layers.object_detection._object_proposal.filter_boxes(proposals, minimum)

    numpy.testing.assert_array_equal(keras.backend.eval(results), numpy.array([0, 1]))
