import keras.engine

import keras_rcnn.backend


class ObjectProposal(keras.engine.topology.Layer):
    """
    Transforms RPN outputs (per-anchor scores and bounding box regression estimates) into object proposals.
    """

    def __init__(self, maximum_proposals=300, **kwargs):
        self.output_dim = (None, None, 4)

        self.maximum_proposals = maximum_proposals

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return keras_rcnn.backend.propose(inputs[0], inputs[1], self.maximum_proposals)

    def compute_output_shape(self, input_shape):
        return self.output_dim
