import keras.engine


class RegionProposalNetwork(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        self.output_dim = (None, None, 4)

        super(RegionProposalNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RegionProposalNetwork, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return self.output_dim
