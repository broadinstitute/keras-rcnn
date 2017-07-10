import keras.layers
import keras.models
import numpy

import keras_rcnn.layers


class TestObjectProposal:
    def test_build(self):
        assert True

    def test_compute_output_shape(self, object_proposal_layer):
        assert object_proposal_layer.compute_output_shape((14, 14)) == (None, None, 4)
