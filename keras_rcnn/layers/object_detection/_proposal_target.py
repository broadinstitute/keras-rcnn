import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend


class ProposalTarget(keras.engine.topology.Layer):
    def __init__(self, allowed_border=0, clobber_positives=False, negative_overlap=0.3, positive_overlap=0.7, **kwargs):
        self.allowed_border    = allowed_border
        self.clobber_positives = clobber_positives
        self.negative_overlap  = negative_overlap
        self.positive_overlap  = positive_overlap

        super(ProposalTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ProposalTarget, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gt_boxes, im_info = inputs

        # TODO: Fix usage of batch index
        shape = im_info[0, :2]
        scale = im_info[0, 2]

        # 1. Generate proposals from bbox deltas and shifted anchors
        all_anchors = keras_rcnn.backend.shift(shape, 16)

        # only keep anchors inside the image
        indices, anchors = keras_rcnn.backend.inside_image(all_anchors, im_info, self.allowed_border)

        # 2. obtain indices of gt boxes with the greatest overlap, balanced labels
        argmax_overlaps_indices, labels = keras_rcnn.backend.label(anchors, gt_boxes, indices, self.negative_overlap, self.clobber_positives)

        gt_boxes = keras.backend.gather(gt_boxes, argmax_overlaps_indices)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        bbox_reg_targets = keras_rcnn.backend.bbox_transform(anchors, gt_boxes)

        return labels, bbox_reg_targets

    def compute_output_shape(self, input_shape):
        return (None,), (None, 4)
