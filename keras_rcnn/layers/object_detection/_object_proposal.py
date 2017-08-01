import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend


class ObjectProposal(keras.engine.topology.Layer):
    def __init__(self, maximum_proposals=300, **kwargs):
        self.output_dim = (None, maximum_proposals, 4)
        # TODO : Parametrize this
        self.min_size   = 16 # minimum width/height of proposals in original image size

        self.maximum_proposals = maximum_proposals

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        im_info, boxes, scores = inputs

        # TODO: Fix usage of batch index
        shape = im_info[0, :2]
        scale = im_info[0, 2]

        # 1. generate proposals from bbox deltas and shifted anchors
        shifted = keras_rcnn.backend.shift(shape, 16)
        proposals = keras.backend.reshape(boxes, (-1, 4))
        proposals = keras_rcnn.backend.bbox_transform_inv(shifted, proposals)

        # 2. clip predicted boxes to image
        proposals = keras_rcnn.backend.clip(proposals, shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        indices = keras_rcnn.backend.filter_boxes(proposals, self.min_size * scale)
        proposals = keras.backend.gather(proposals, indices)
        scores = scores[..., (scores.shape[-1] // 2):]
        scores = keras.backend.reshape(scores, (-1, 1))
        scores = keras.backend.gather(scores, indices)
        scores = keras.backend.flatten(scores)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        # TODO : Needs to be implemented?

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        indices = keras_rcnn.backend.non_maximum_suppression(proposals, scores, self.maximum_proposals, 0.7)

        proposals = keras.backend.gather(proposals, indices)

        return keras.backend.expand_dims(proposals, 0)

    def compute_output_shape(self, input_shape):
        return self.output_dim
