import keras.backend
import keras.engine

import keras_rcnn.backend


class ProposalTarget(keras.layers.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.

    # Arguments
    fg_fraction: percent foreground objects
    batchsize: number of objects in a batch
    num_images: number of images to consider per batch (set to 1 for the time being)
    num_classes: number of classes (object+background)

    # Input shape
    (None, None, 4), (None, None, classes), (None, None, 4)

    # Output shape
    [(None, None, 4), (None, None, classes), (None, None, 4)]
    """

    def __init__(self, fg_fraction=0.5, fg_thresh=0.7, bg_thresh_hi=0.5, bg_thresh_lo=0.1, batchsize=256, num_images=2, **kwargs):
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.batchsize = batchsize
        self.num_classes = None
        self.num_images = num_images
        super(ProposalTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ProposalTarget, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Proposal ROIs (x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2)
        # and other times after box coordinates -- normalize to one format

        # labels (class1, class2, ... , num_classes)
        # Include ground-truth boxes in the set of candidate rois
        proposals, bounding_boxes, labels = inputs

        proposals = keras.backend.concatenate((proposals, bounding_boxes), axis=1)

        rois_per_image = self.batchsize / self.num_images
        fg_rois_per_image = keras.backend.round(self.fg_fraction * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets

        # TODO: Fix usage of batch index
        batch_index = 0
        proposals = proposals[batch_index, :, :]
        bounding_boxes = bounding_boxes[batch_index, :, :]
        labels = labels[batch_index, :, :]
        rois, labels, bbox_targets = keras_rcnn.backend.sample_rois(proposals, bounding_boxes, labels, fg_rois_per_image, rois_per_image, self.fg_thresh, self.bg_thresh_hi, self.bg_thresh_lo)
        self.proposals = keras.backend.shape(rois)[0]
        return [keras.backend.expand_dims(rois, axis=0), keras.backend.expand_dims(labels, axis=0), keras.backend.expand_dims(bbox_targets, axis=0)]

    def compute_output_shape(self, input_shape):
        num_classes = input_shape[2][2]
        return [(1, None, 4), (1, None, num_classes), (1, None, 4)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None]
