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
        scores, gt_boxes, im_info = inputs

        # TODO: Fix usage of batch index
        shape = im_info[0, :2]
        scale = im_info[0, 2]

        # 1. Generate proposals from bbox deltas and shifted anchors
        all_anchors = keras_rcnn.backend.shift(shape, 16)

        # only keep anchors inside the image
        indices, anchors = keras_rcnn.backend.inside_image(all_anchors, im_info, self.allowed_border)

        # label: 1 is positive, 0 is negative, -1 is dont care
        ones   = keras.backend.ones_like(indices, dtype=keras.backend.floatx())
        zeros  = keras.backend.zeros_like(indices, dtype=keras.backend.floatx())
        labels = ones * -1

        argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = keras_rcnn.backend.overlapping(anchors, gt_boxes, indices)

        if not self.clobber_positives:
            labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, self.negative_overlap), zeros, labels)

        # fg label: for each gt, anchor with highest overlap
        #TODO Find Keras equivalent of this
        #labels[gt_argmax_overlaps_inds] = 1

        # fg label: above threshold IOU
        labels = keras_rcnn.backend.where(keras.backend.greater_equal(max_overlaps, self.positive_overlap), ones, labels)

        if self.clobber_positives:
            labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, self.negative_overlap), zeros, labels)

        # subsample positive labels if we have too many
        # subsample negative labels if we have too many
        # map up to original set of anchors

        return anchors

    def compute_output_shape(self, input_shape):
        return (None, 4)
