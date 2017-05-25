import keras.engine.topology

import keras_rcnn.backend


class Anchor(keras.engine.topology.Layer):
    """
    Assign anchors to ground-truth targets.
    
    It produces:
        1. anchor classification labels
        2. bounding-box regression targets.
    """
    def __init__(self, stride=16, scales=None, ratios=None, **kwargs):
        if ratios is None:
            self.ratios = [0.5, 1, 2]
        else:
            self.ratios = ratios

        if scales is None:
            scales = [8, 16, 32]
        else:
            self.scales = scales

        self.output_dim = (None, None, 4)

        self.ratios = ratios

        self.scales = scales

        self.stride = stride

        super(Anchor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Anchor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return self.output_dim


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
