# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend


class ProposalTarget(keras.layers.Layer):
    """
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
    def __init__(self, foreground=0.5, foreground_threshold=(0.5, 1.0), background_threshold=(0.1, 0.5), maximum_proposals=32, **kwargs):
        """
        :param foreground:
        :param foreground_threshold:
        :param background_threshold:
        :param maximum_proposals:
        """
        self._batch_size = None

        self.foreground = foreground

        self.foreground_threshold = foreground_threshold
        self.background_threshold = background_threshold

        self.maximum_proposals = maximum_proposals

        self.rois_per_image = self.maximum_proposals / self.batch_size
        self.fg_rois_per_image = keras.backend.cast(self.foreground * self.rois_per_image, 'int32')

        self.fg_rois_per_this_image = None

        super(ProposalTarget, self).__init__(**kwargs)

    @property
    def batch_size(self):
        if self._batch_size:
            return self._batch_size
        else:
            self._batch_size = 1

            return self._batch_size

    @batch_size.setter
    def batch_size(self, x):
        self._batch_size = x

    def build(self, input_shape):
        super(ProposalTarget, self).build(input_shape)

    def call(self, inputs, training=None):
        # Proposal ROIs (x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2)
        # and other times after box coordinates -- normalize to one format

        # target_categories (class1, class2, ... , num_classes)
        # Include ground-truth boxes in the set of candidate rois
        target_bounding_boxes, target_categories, output_proposal_bounding_boxes = inputs

        output_proposal_bounding_boxes = keras.backend.in_train_phase(
            x=keras.backend.concatenate((output_proposal_bounding_boxes, target_bounding_boxes), axis=1),
            alt=output_proposal_bounding_boxes,
            training=training
        )

        # Sample rois with classification target_categories and bounding box regression
        # targets

        # TODO: Fix usage of batch index
        batch_index = 0

        output_proposal_bounding_boxes = output_proposal_bounding_boxes[batch_index, :, :]
        target_bounding_boxes = target_bounding_boxes[batch_index, :, :]
        target_categories = target_categories[batch_index, :, :]

        # TODO: Fix hack
        condition = keras.backend.not_equal(keras.backend.sum(target_bounding_boxes), 0)

        def test(proposals, gt_boxes, gt_labels):
            N = keras.backend.shape(proposals)[0]
            number_of_classes = keras.backend.shape(gt_labels)[1]
            number_of_coordinates = 4 * number_of_classes
            return proposals, tensorflow.zeros((N, number_of_classes)), tensorflow.zeros((N, number_of_coordinates))

        sample_outputs = keras.backend.switch(condition,
                                              lambda: self.sample(output_proposal_bounding_boxes, target_bounding_boxes, target_categories),
                                              lambda: test(output_proposal_bounding_boxes, target_bounding_boxes, target_categories))

        output_proposal_bounding_boxes = keras.backend.expand_dims(sample_outputs[0], 0)
        target_categories = keras.backend.expand_dims(sample_outputs[1], 0)
        bounding_box_targets = keras.backend.expand_dims(sample_outputs[2], 0)

        return [bounding_box_targets, target_categories, output_proposal_bounding_boxes]

    def get_config(self):
        configuration = {
            "background_threshold": self.background_threshold,
            "foreground": self.foreground,
            "foreground_threshold": self.foreground_threshold,
            "maximum_proposals": self.maximum_proposals
        }

        return {**super(ProposalTarget, self).get_config(), **configuration}

    def sample(self, proposals, true_bounding_boxes, true_labels):
        """
        Generate a random sample of RoIs comprising foreground and background
        examples.

        all_rois is (N, 4)
        gt_boxes is (K, 4) with 4 coordinates

        gt_labels is in one hot form
        """
        number_of_classes = keras.backend.shape(true_labels)[1]

        true_labels = keras.backend.argmax(true_labels, axis=1)

        intersection_over_union = keras_rcnn.backend.intersection_over_union(proposals, true_bounding_boxes)

        gt_assignment = keras.backend.argmax(intersection_over_union, axis=1)

        maximum_intersection_over_union = keras.backend.max(intersection_over_union, axis=1)

        # finds the ground truth labels corresponding to the ground truth boxes with greatest overlap for each predicted regions of interest
        # TODO: rename `all_labels`
        all_labels = keras.backend.gather(true_labels, gt_assignment)

        # Select proposals with given parameters for fg/bg objects
        # TODO: rename `find_foreground_and_background_proposal_indices`
        # TODO: rename `foreground_and_background_proposal_indices`

        foreground_and_background_proposal_indices = self.find_foreground_and_background_proposal_indices(maximum_intersection_over_union)

        # Select sampled values from various arrays:
        sampled_labels = keras.backend.gather(all_labels, foreground_and_background_proposal_indices)
        sampled_proposal_bounding_boxes = keras.backend.gather(proposals, foreground_and_background_proposal_indices)

        sampled_labels = self.set_label_background(sampled_labels)

        true_bounding_boxes = keras.backend.gather(true_bounding_boxes, keras.backend.gather(gt_assignment, foreground_and_background_proposal_indices))

        bbox_targets = self.get_bbox_targets(sampled_proposal_bounding_boxes, true_bounding_boxes, sampled_labels, number_of_classes)

        sampled_labels = keras.backend.one_hot(sampled_labels, number_of_classes)

        return sampled_proposal_bounding_boxes, sampled_labels, bbox_targets

    def set_label_background(self, labels):
        # Clamp labels for the background RoIs to 0
        update_indices = keras.backend.arange(self.fg_rois_per_this_image, keras.backend.shape(labels)[0])
        update_indices = keras.backend.reshape(update_indices, (-1, 1))

        # By making the label = background
        inverse_labels = keras_rcnn.backend.gather_nd(labels, update_indices) * -1
        labels = keras_rcnn.backend.scatter_add_tensor(labels, update_indices, inverse_labels)

        return labels

    def compute_output_shape(self, input_shape):
        num_classes = input_shape[1][2]

        self.batch_size = input_shape[0][0]

        return [(self.batch_size, None, 4 * num_classes), (self.batch_size, None, num_classes), (self.batch_size, None, 4)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None]

    def get_bbox_targets(self, rois, gt_boxes, labels, num_classes):
        gt_boxes = keras.backend.cast(gt_boxes, keras.backend.floatx())
        targets = keras_rcnn.backend.bbox_transform(
            rois,
            gt_boxes
        )
        return self.get_bbox_regression_labels(targets, labels, num_classes)

    def find_foreground_and_background_proposal_indices(self, max_overlaps):
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = keras_rcnn.backend.where((max_overlaps <= self.foreground_threshold[1]) & (max_overlaps >= self.foreground_threshold[0]))

        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        self.fg_rois_per_this_image = keras.backend.minimum(self.fg_rois_per_image, keras.backend.shape(fg_inds)[0])

        # Sample foreground regions without replacement
        fg_inds = self.sample_indices(fg_inds, self.fg_rois_per_this_image)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = keras_rcnn.backend.where((max_overlaps < self.background_threshold[1]) & (max_overlaps >= self.background_threshold[0]))

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

    def sample_indices(self, indices, size):
        return keras_rcnn.backend.shuffle(keras.backend.reshape(indices, (-1,)))[:size]

    def get_bbox_regression_labels(self, bbox_target_data, labels, num_classes):
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

        updates = keras.backend.cast(updates, keras.backend.floatx())
        bbox_targets = keras_rcnn.backend.scatter_add_tensor(bbox_targets, indices, updates)

        return bbox_targets
