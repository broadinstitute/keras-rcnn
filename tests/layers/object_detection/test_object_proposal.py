import keras.backend
import numpy

import keras_rcnn.backend
import keras_rcnn.layers


class TestObjectProposal:
    def test_call(self):
        metadata = keras.backend.variable([[224, 224, 1.5]])

        deltas = numpy.random.random((1, 14, 14, 9 * 4))
        scores = numpy.random.random((1, 14, 14, 9 * 2))
        anchors = numpy.zeros((1, 14 * 14 * 9, 4)).astype('float32')

        deltas = keras.backend.variable(deltas)
        scores = keras.backend.variable(scores)

        object_proposal = keras_rcnn.layers.ObjectProposal()

        object_proposal.call([anchors, metadata, deltas, scores])


def test_filter_boxes():
    proposals = numpy.array(
        [[0, 2, 3, 10],
         [-1, -5, 4, 8],
         [0, 0, 1, 1]]
    )

    minimum = 3

    results = keras_rcnn.layers.object_detection._object_proposal.filter_boxes(
        proposals, minimum)

    numpy.testing.assert_array_equal(keras.backend.eval(results),
                                     numpy.array([0, 1]))
