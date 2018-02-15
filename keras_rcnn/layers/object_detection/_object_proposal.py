# -*- coding: utf-8 -*-

import keras.backend
import keras.engine

import keras_rcnn.backend
import keras_rcnn.layers


class ObjectProposal(keras.engine.topology.Layer):
    """Propose object-containing regions from anchors

    # Arguments
        maximum_proposals: maximum number of regions allowed
        min_size: minimum width/height of proposals
        stride: stride size

    # Input shape
        (width of feature map, height of feature map, scale), (None, 4), (None)

    # Output shape
        (# images, # proposals, 4)
    """
    def __init__(self, maximum_proposals=300, minimum_size=16, stride=16, **kwargs):
        self.maximum_proposals = maximum_proposals

        # minimum width/height of proposals in original image size
        self.minimum_size = minimum_size

        self.stride = stride

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        `image_shape_and_scale` has the shape [width, height, scale]
        """
        anchors, image_shape_and_scale, deltas, scores = inputs
        anchors = keras.backend.reshape(anchors, (-1, 4))

        # TODO: Fix usage of batch index
        batch_index = 0

        image_shape = image_shape_and_scale[batch_index, :2]
        image_scale = image_shape_and_scale[batch_index, -1]

        # 1. generate proposals from bbox deltas and shifted anchors

        deltas = keras.backend.reshape(deltas, (-1, 4))
        scores = keras.backend.reshape(scores, (-1, 1))

        deltas = keras_rcnn.backend.bbox_transform_inv(anchors, deltas)

        # 2. clip predicted boxes to image
        proposals = keras_rcnn.backend.clip(deltas, image_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        indices = filter_boxes(proposals, self.minimum_size * image_scale)
        proposals = keras.backend.gather(proposals, indices)

        scores = scores[..., (scores.shape[-1] // 2):]
        scores = keras.backend.reshape(scores, (-1, 1))
        scores = keras.backend.gather(scores, indices)
        scores = keras.backend.flatten(scores)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        indices = keras_rcnn.backend.argsort(scores)

        # TODO: is this a sensible value? parameterize?
        rpn_pre_nms_top_n = 12000

        # 5. take top pre_nms_topN (e.g. 6000)
        if rpn_pre_nms_top_n > 0:
            indices = indices[:rpn_pre_nms_top_n]

        proposals = keras.backend.gather(proposals, indices)
        scores = keras.backend.gather(scores, indices)

        # 6. apply nms (e.g. threshold = 0.7)
        indices = keras_rcnn.backend.non_maximum_suppression(boxes=proposals, scores=scores, maximum=self.maximum_proposals, threshold=0.7)

        proposals = keras.backend.gather(proposals, indices)

        # 8. return the top proposals (-> RoIs top)
        return keras.backend.expand_dims(proposals, 0)

    def compute_output_shape(self, input_shape):
        return None, None, 4

    def get_config(self):
        configuration = {
            "maximum_proposals": self.maximum_proposals,
            "minimum_size": self.minimum_size,
            "stride": self.stride
        }

        return {**super(ObjectProposal, self).get_config(), **configuration}


def filter_boxes(proposals, minimum):
    """
    Filters proposed RoIs so that all have width and height at least as big as
    minimum
    """
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indices = keras_rcnn.backend.where((ws >= minimum) & (hs >= minimum))

    indices = keras.backend.flatten(indices)

    return keras.backend.cast(indices, "int32")
