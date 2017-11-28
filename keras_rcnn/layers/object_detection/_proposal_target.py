# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend


class ProposalTarget(keras.layers.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces
    proposal classification labels and bounding-box regression targets.

    # Arguments
    fg_fraction: percent foreground objects

    batchsize: number of objects in a batch

    num_images: number of images to consider per batch (set to 1 for the
    time being)

    num_classes: number of classes (object+background)

    # Input shape
    (None, None, 4), (None, None, classes), (None, None, 4)

    # Output shape
    [(None, None, 4), (None, None, classes), (None, None, 4)]
    """

    def __init__(self, fg_fraction=0.25, fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.1, batchsize=256, num_images=1, **kwargs):
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.batchsize = batchsize
        self.num_images = num_images
        self.rois_per_image = self.batchsize / self.num_images
        self.fg_rois_per_image = keras.backend.round(self.fg_fraction * self.rois_per_image)
        self.fg_rois_per_this_image = None

        super(ProposalTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ProposalTarget, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # Proposal ROIs (x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2)
        # and other times after box coordinates -- normalize to one format

        # labels (class1, class2, ... , num_classes)
        # Include ground-truth boxes in the set of candidate rois
        proposals, labels, bounding_boxes = inputs

        proposals = keras.backend.concatenate((proposals, bounding_boxes),
                                              axis=1)

        # Sample rois with classification labels and bounding box regression
        # targets

        # TODO: Fix usage of batch index
        batch_index = 0
        proposals = proposals[batch_index, :, :]
        bounding_boxes = bounding_boxes[batch_index, :, :]
        labels = labels[batch_index, :, :]

        sample_outputs = self.sample_rois(proposals, bounding_boxes, labels)

        rois = keras.backend.expand_dims(sample_outputs[0], 0)
        labels = keras.backend.expand_dims(sample_outputs[1], 0)
        bbox_targets = keras.backend.expand_dims(sample_outputs[2], 0)

        return [rois, labels, bbox_targets]

    def get_config(self):
        configuration = {
            "fg_fraction": self.fg_fraction,
            "fg_thresh": self.fg_thresh,
            "bg_thresh_hi": self.bg_thresh_hi,
            "bg_thresh_lo": self.bg_thresh_lo,
            "batchsize": self.batchsize,
            "num_images": self.num_images
        }

        return {**super(ProposalTarget, self).get_config(), **configuration}

    def sample_rois(self, all_rois, gt_boxes, gt_labels):
        """
        Generate a random sample of RoIs comprising foreground and background
        examples.

        all_rois is (N, 4)
        gt_boxes is (K, 4) with 4 coordinates

        gt_labels is in one hot form
        """
        num_classes = keras.backend.shape(gt_labels)[1]
        gt_labels = keras.backend.argmax(gt_labels, axis=1)

        # overlaps: (rois x gt_boxes)
        # finds the overlapping regions between the predicted regions of interests and ground truth bounding boxes
        overlaps = keras_rcnn.backend.overlap(all_rois, gt_boxes)
        gt_assignment = keras.backend.argmax(overlaps, axis=1)
        max_overlaps = keras.backend.max(overlaps, axis=1)

        # finds the ground truth labels corresponding to the ground truth boxes with greatest overlap for each predicted regions of interest
        labels = keras.backend.gather(gt_labels, gt_assignment)

        # Select RoIs with given parameters for fg/bg objects
        keep_inds = self.get_fg_bg_rois(max_overlaps)

        # Select sampled values from various arrays:
        labels = keras.backend.gather(labels, keep_inds)
        rois = keras.backend.gather(all_rois, keep_inds)

        labels = self.set_label_background(labels)

        # Compute bounding-box regression targets for an image.
        gt_boxes = keras.backend.gather(gt_boxes, keras.backend.gather(gt_assignment, keep_inds))
        bbox_targets = self.get_bbox_targets(rois, gt_boxes, labels, num_classes)

        labels = keras.backend.one_hot(labels, num_classes)
        return rois, labels, bbox_targets

    def set_label_background(self, labels):
        # Clamp labels for the background RoIs to 0
        update_indices = keras.backend.arange(self.fg_rois_per_this_image, keras.backend.shape(labels)[0])
        update_indices = keras.backend.reshape(update_indices, (-1, 1))

        # By making the label = background
        inverse_labels = keras_rcnn.backend.gather_nd(labels, update_indices) * -1
        labels = keras_rcnn.backend.scatter_add_tensor(labels, update_indices, inverse_labels)

        return labels

    def compute_output_shape(self, input_shape):
        num_classes = input_shape[2][2]
        return [(1, None, 4), (1, None, num_classes), (1, None, 4)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None]

    def get_bbox_targets(self, rois, gt_boxes, labels, num_classes):
        gt_boxes = keras.backend.cast(gt_boxes, keras.backend.floatx())
        targets = keras_rcnn.backend.bbox_transform(
            rois,
            gt_boxes
        )
        return get_bbox_regression_labels(targets, labels, num_classes)


    def get_fg_bg_rois(self, max_overlaps):

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = keras_rcnn.backend.where(max_overlaps >= self.fg_thresh)

        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_image = keras.backend.cast(self.fg_rois_per_image, 'int32')
        self.fg_rois_per_this_image = keras.backend.minimum(fg_rois_per_image, keras.backend.shape(fg_inds)[0])

        # Sample foreground regions without replacement
        fg_inds = self.sample_indices(fg_inds, self.fg_rois_per_this_image)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = keras_rcnn.backend.where((max_overlaps < self.bg_thresh_hi) & (max_overlaps >= self.bg_thresh_lo))

        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = keras.backend.cast(self.rois_per_image, 'int32') - self.fg_rois_per_this_image
        bg_rois_per_this_image = keras.backend.cast(bg_rois_per_this_image, 'int32')
        bg_rois_per_this_image = keras.backend.minimum(bg_rois_per_this_image, keras.backend.shape(bg_inds)[0])

        # Sample background regions without replacement
        bg_inds = self.sample_indices(bg_inds, bg_rois_per_this_image)

        # The indices that we're selecting (both fg and bg)
        keep_inds = keras.backend.concatenate([fg_inds, bg_inds])

        return keep_inds

    def sample_indices(self, indices, threshold):
        def no_sample(indices):
            return keras.backend.reshape(indices, (-1,))

        def sample(indices, size):
            return keras_rcnn.backend.shuffle(keras.backend.reshape(indices, (-1,)))[:size]

        return keras.backend.switch(keras.backend.shape(indices)[0] > 0, lambda: no_sample(indices), lambda: sample(indices, threshold))


def get_bbox_regression_labels(bbox_target_data, labels, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    form N x (tx, ty, tw, th), labels N
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target: N x 4K blob of regression targets
    """

    n = keras.backend.shape(bbox_target_data)[0]

    bbox_targets = tensorflow.zeros((n, 4 * num_classes), dtype=keras.backend.floatx())

    inds = keras.backend.reshape(keras_rcnn.backend.where(labels > 0), (-1,))

    labels = keras.backend.gather(labels, inds)

    start = 4 * labels

    ii = keras.backend.expand_dims(inds)
    ii = keras.backend.tile(ii, [4, 1])

    aa = keras.backend.expand_dims(keras.backend.concatenate([start, start + 1, start + 2, start + 3], 0))
    aa = keras.backend.cast(aa, dtype='int64')

    indices = keras.backend.concatenate([ii, aa], 1)

    updates = keras.backend.gather(bbox_target_data, inds)
    updates = keras.backend.transpose(updates)
    updates = keras.backend.reshape(updates, (-1,))

    bbox_targets = keras_rcnn.backend.scatter_add_tensor(bbox_targets, indices, updates)

    return bbox_targets
