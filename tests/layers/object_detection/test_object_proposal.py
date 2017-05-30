import numpy


class TestObjectProposal:
    def test_build(self):
        assert True

    def test_call(self, object_proposal_model):
        object_proposal_model.compile("sgd", "mse")

        a = numpy.random.rand(1, 14, 14, 4 * 9)
        b = numpy.random.rand(1, 14, 14, 2 * 9)

        prediction = object_proposal_model.predict([a, b])

        assert prediction.shape == (1, 100, 4)

    def test_compute_output_shape(self, object_proposal_layer):
        assert object_proposal_layer.compute_output_shape((14, 14)) == (None, None, 4)
