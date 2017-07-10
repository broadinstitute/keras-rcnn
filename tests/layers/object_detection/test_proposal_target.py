import keras.backend
import keras.utils
import numpy

import keras_rcnn.layers


class TestProposalTarget:
    def test_call(self):
        proposal_target = keras_rcnn.layers.ProposalTarget()

        proposals = numpy.random.random((1, 300, 4))
        proposals = keras.backend.variable(proposals)

        bounding_boxes = numpy.random.choice(range(0, 224), (1, 10, 4))
        bounding_boxes = keras.backend.variable(bounding_boxes)

        labels = numpy.random.choice(range(0, 2), (1, 10))
        labels = keras.utils.to_categorical(labels)
        labels = keras.backend.variable(labels)
        labels = keras.backend.expand_dims(labels, 0)

        proposal_target.call([proposals, bounding_boxes, labels])
