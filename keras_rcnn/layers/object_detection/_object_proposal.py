import keras.backend
import keras.engine
import numpy

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

        rr = keras.backend.shape(scores)[1]
        cc = keras.backend.shape(scores)[2]

        # TODO: Fix usage of batch index
        batch_index = 0

        image_shape = image_shape_and_scale[batch_index, :2]
        image_scale = image_shape_and_scale[batch_index, -1]

        # 1. generate proposals from bbox deltas and shifted anchors
        anchors = keras_rcnn.backend.shift([rr, cc], self.stride)

        deltas = keras.backend.reshape(deltas, (-1, 4))
        scores = keras.backend.reshape(scores, (-1, 1))

        deltas = bbox_transform_inv(anchors, deltas)

        # 2. clip predicted boxes to image
        proposals = keras_rcnn.backend.clip(deltas, image_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        indices = filter_boxes(proposals, self.min_size * image_scale)
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
        indices = keras_rcnn.backend.non_maximum_suppression(proposals, scores, self.maximum_proposals, 0.7)

        proposals = keras.backend.gather(proposals, indices)

        # 8. return the top proposals (-> RoIs top)
        return keras.backend.expand_dims(proposals, 0)

    def compute_output_shape(self, input_shape):
        return None, self.maximum_proposals, 4


def bbox_transform_inv(shifted, boxes):
    def shape_zero():
        x = keras.backend.int_shape(boxes)[-1]

        return keras.backend.zeros_like(x, dtype=keras.backend.floatx())

    def shape_non_zero():
        a = shifted[:, 2] - shifted[:, 0] + 1.0
        b = shifted[:, 3] - shifted[:, 1] + 1.0

        ctr_x = shifted[:, 0] + 0.5 * a
        ctr_y = shifted[:, 1] + 0.5 * b

        dx = boxes[:, 0::4]
        dy = boxes[:, 1::4]
        dw = boxes[:, 2::4]
        dh = boxes[:, 3::4]

        pred_ctr_x = dx * a[:, keras_rcnn.backend.newaxis] + ctr_x[:, keras_rcnn.backend.newaxis]
        pred_ctr_y = dy * b[:, keras_rcnn.backend.newaxis] + ctr_y[:, keras_rcnn.backend.newaxis]

        pred_w = keras.backend.exp(dw) * a[:, keras_rcnn.backend.newaxis]
        pred_h = keras.backend.exp(dh) * b[:, keras_rcnn.backend.newaxis]

        prediction = [
            pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h
        ]

        return keras.backend.concatenate(prediction)

    zero_boxes = keras.backend.equal(keras.backend.shape(boxes)[0], 0)

    pred_boxes = keras.backend.switch(zero_boxes, shape_zero, shape_non_zero)

    return pred_boxes


def filter_boxes(proposals, minimum):
    """
    Filters proposed RoIs so that all have width and height at least as big as minimum

    """
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indices = keras_rcnn.backend.where((ws >= minimum) & (hs >= minimum))

    indices = keras.backend.flatten(indices)

    return keras.backend.cast(indices, "int32")
