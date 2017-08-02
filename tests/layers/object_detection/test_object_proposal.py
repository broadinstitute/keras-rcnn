import keras.backend
import numpy

import keras_rcnn.layers


class TestObjectProposal:
    def test_call(self):
        image_shape_and_scale = keras.backend.variable([[224, 224, 1.5]])

        objectness_scores = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 2)))
        reference_bounding_boxes = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 4)))

        object_proposal = keras_rcnn.layers.ObjectProposal()

        object_proposal.call([image_shape_and_scale, reference_bounding_boxes, objectness_scores])
