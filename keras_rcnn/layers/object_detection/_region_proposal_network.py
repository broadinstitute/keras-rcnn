import keras.engine


class RegionProposalNetwork(keras.engine.topology.Layer):
    def __init__(self, proposals=300, ratios=None, scales=None, stride=16, **kwargs):
        self.bounding_boxes = None

        self._image_features = None

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
    def image_features(self):
        return self._image_features

    @image_features.setter
    def image_features(self, value):
        self._image_features = keras.backend.conv2d(value, 512, (3, 3), "same")

        self._image_features = keras.backend.relu(self.image_features)

    @property
    def n_anchors(self):
        return len(self.ratios) * len(self.scales)

    def build(self, input_shape):
        super(RegionProposalNetwork, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.bounding_boxes, self.image_features = inputs

        rpn_cls_score = self.classifier()

        rpn_bbox_pred = self.regressor()

        return inputs

    def classifier(self):
        x = keras.backend.conv2d(self.image_features, self.n_anchors * 2, (1, 1))

        return keras.backend.sigmoid(x)

    def compute_output_shape(self, input_shape):
        return (None, self.proposals, 4), (None, self.proposals, 1)

    def regressor(self):
        x = keras.backend.conv2d(self.image_features, self.n_anchors * 4, (1, 1))

        return keras.backend.sigmoid(x)
