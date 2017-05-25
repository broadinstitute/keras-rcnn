import keras.engine.topology

import keras_rcnn.backend


class ObjectProposal(keras.engine.topology.Layer):
    """
    Transforms RPN outputs (per-anchor scores and bounding box regression estimates) into object proposals.
    """
    def __init__(self, proposals, **kwargs):
        self.output_dim = (None, None, 4)

        self.proposals = proposals

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return keras_rcnn.backend.propose(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return self.output_dim
