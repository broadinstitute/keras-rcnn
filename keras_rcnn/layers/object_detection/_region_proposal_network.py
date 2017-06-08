import keras.engine


class RegionProposalNetwork(keras.engine.topology.Layer):
    def __init__(self, proposals=300, ratios=None, scales=None, stride=16, **kwargs):
        self.bounding_boxes = None

        self.image_features = None

        self.proposals = proposals

        if ratios is None:
            self.ratios = [0.5, 1, 2]
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = [8, 16, 32]
        else:
            self.scales = scales

        self.stride = stride

        super(RegionProposalNetwork, self).__init__(**kwargs)

    @property
    def n_anchors(self):
        return len(self.ratios) * len(self.scales)

    def build(self, input_shape):
        super(RegionProposalNetwork, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.bounding_boxes, self.image_features = inputs

        return inputs

    def compute_output_shape(self, input_shape):
        return (None, self.proposals, 4), (None, self.proposals, 1)
