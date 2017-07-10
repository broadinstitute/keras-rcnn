import keras.backend
import keras.engine

import keras_rcnn.backend

# FIXME: remove global
RPN_PRE_NMS_TOP_N = 12000


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
    def __init__(self, maximum_proposals=300, min_size=16, stride=16, **kwargs):
        self.maximum_proposals = maximum_proposals

        # minimum width/height of proposals in original image size
        self.min_size = min_size

        self.stride = stride

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        `image_shape_and_scale` has the shape [width, height, scale]
        """
        image_shape_and_scale, deltas, scores = inputs

        # the first set of anchors channels are bg probs
        # the second set are the fg probs, which we want
        # scores = scores[:, :, :, 9:]

        rr, cc = keras.backend.int_shape(scores)[1:-1]

        # TODO: Fix usage of batch index
        batch_index = 0

        image_shape = image_shape_and_scale[batch_index, :2]
        image_scale = image_shape_and_scale[batch_index, -1]

        # 1. generate proposals from bbox deltas and shifted anchors
        anchors = keras_rcnn.backend.shift([rr, cc], self.stride)

        deltas = keras.backend.reshape(deltas, (-1, 4))
        scores = keras.backend.reshape(scores, (-1, 1))

        deltas = keras_rcnn.backend.bbox_transform_inv(anchors, deltas)

        # 2. clip predicted boxes to image
        proposals = keras_rcnn.backend.clip(deltas, image_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        indices = keras_rcnn.backend.filter_boxes(proposals, self.min_size * image_scale)
        proposals = keras.backend.gather(proposals, indices)

        scores = scores[..., (scores.shape[-1] // 2):]
        scores = keras.backend.reshape(scores, (-1, 1))
        scores = keras.backend.gather(scores, indices)
        scores = keras.backend.flatten(scores)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        indices = keras_rcnn.backend.argsort(scores)

        # 5. take top pre_nms_topN (e.g. 6000)
        if RPN_PRE_NMS_TOP_N > 0:
            indices = indices[:RPN_PRE_NMS_TOP_N]

        proposals = keras.backend.gather(proposals, indices)
        scores = keras.backend.gather(scores, indices)

        # 6. apply nms (e.g. threshold = 0.7)
        indices = keras_rcnn.backend.non_maximum_suppression(proposals, scores, self.maximum_proposals, 0.7)

        proposals = keras.backend.gather(proposals, indices)
        scores = keras.backend.gather(scores, indices)

        # 8. return the top proposals (-> RoIs top)
        return keras.backend.expand_dims(proposals, 0)

    def compute_output_shape(self, input_shape):
        return None, self.maximum_proposals, 4
