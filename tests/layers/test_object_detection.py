import keras.layers
import numpy

import keras_rcnn.layers.object_detection


class TestAnchor:
    def test_build(self):
        assert True

    def test_call(self):
        assert True

    def test_compute_output_shape(self):
        assert True


class TestObjectProposal:
    def test_build(self):
        assert True

    def test_call(self, object_proposal):
        object_proposal.compile("sgd", "mse")

        a = numpy.random.rand(1, 14, 14, 4 * 9)
        b = numpy.random.rand(1, 14, 14, 2 * 9)

        prediction = object_proposal.predict([a, b])

        assert prediction.shape == (1, 100, 4)

    def test_compute_output_shape(self, object_proposal):
        layer = object_proposal.layers[-1]

        assert layer.compute_output_shape((14, 14)) == (None, None, 4)
