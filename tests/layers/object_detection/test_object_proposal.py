import keras.backend


class TestObjectProposal:
    def test_compute_output_shape(self, object_proposal_layer):
        assert object_proposal_layer.compute_output_shape((14, 14)) == (None, 300, 4)

    def test_propose_objects(self, object_proposal_layer):
        a = keras.backend.zeros((1, 14, 14, 9 * 4))
        b = keras.backend.zeros((1, 14, 14, 9 * 1))

        proposals = object_proposal_layer.propose(a, b, 100)

        assert keras.backend.eval(proposals).shape == (1, 100, 4)
