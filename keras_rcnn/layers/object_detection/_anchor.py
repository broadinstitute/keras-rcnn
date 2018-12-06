# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers


class Anchor(keras.layers.Layer):
    def __init__(
            self,
            aspect_ratios=None,
            base_size=16,
            clobber_positives=False,
            negative_overlap=0.3,
            padding=0,
            positive_overlap=0.7,
            scales=None,
            stride=16,
            **kwargs
    ):
        if aspect_ratios is None:
            aspect_ratios = [0.5, 1, 2]  # [1:2, 1:1, 2:1]

        if scales is None:
            scales = [1, 2, 4, 8, 16]  # [32^{2}, 64^{2}, 128^{2}, 256^{2}, 512^{2}]

        self.padding = padding

        self.r = None
        self.c = None

        self.clobber_positives = clobber_positives

        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap

        self.stride = stride

        self.base_size = base_size

        self.aspect_ratios = keras.backend.variable(aspect_ratios)

        self.scales = keras.backend.variable(scales)

        self.__shifted_anchors = None

        self.metadata = None

        super(Anchor, self).__init__(**kwargs)

    @property
    def _shifted_anchors(self):
        if self.__shifted_anchors:
            return self.__shifted_anchors
        else:
            self.__shifted_anchors = keras_rcnn.backend.shift(
                (self.r, self.c),
                self.stride,
                self.base_size,
                self.aspect_ratios,
                self.scales
            )

            return self.__shifted_anchors

    def build(self, input_shape):
        super(Anchor, self).build(input_shape)

    # TODO: should AnchorTarget only be enabled during training
    def call(self, inputs, **kwargs):
        target_bounding_boxes, metadata, scores = inputs

        self.metadata = metadata[0, :]

        target_bounding_boxes = target_bounding_boxes[0]

        self.r = keras.backend.shape(scores)[1]
        self.c = keras.backend.shape(scores)[2]

        self.k = self.r * self.c * keras.backend.shape(scores)[3]

        # 1. Generate proposals from bbox deltas and shifted anchors
        output_bounding_boxes = self._shifted_anchors

        # only keep anchors inside the image
        indices_inside, anchors = self._inside_image(output_bounding_boxes)

        anchors = keras_rcnn.backend.clip(anchors, self.metadata[:2])

        # 2. obtain indices of gt boxes with the greatest overlap, balanced
        # target_categories
        argmax_overlaps_indices, target_categories = self._label(target_bounding_boxes, anchors, indices_inside)

        target_bounding_boxes = keras.backend.gather(target_bounding_boxes, argmax_overlaps_indices)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        target_bounding_boxes = keras_rcnn.backend.bbox_transform(anchors, target_bounding_boxes)

        # TODO: Why is target_bounding_box_targets' shape (5, ?, 4)? Why is target_bounding_boxes'
        # shape (None, None, 4) and not (None, 4)?
        target_bounding_boxes = keras.backend.reshape(target_bounding_boxes, (-1, 4))

        # map up to original set of anchors
        target_categories = self._unmap(target_categories, indices_inside, fill=-1)

        target_bounding_boxes = self._unmap(target_bounding_boxes, indices_inside, fill=0)

        target_categories = keras.backend.expand_dims(target_categories, axis=0)

        target_bounding_boxes = keras.backend.expand_dims(target_bounding_boxes, axis=0)

        output_bounding_boxes = keras.backend.expand_dims(output_bounding_boxes, axis=0)

        # TODO: implement inside and outside weights
        return [output_bounding_boxes, target_bounding_boxes, target_categories]

    def compute_output_shape(self, input_shape):
        return [(1, None, 4), (1, None, 4), (1, None)]

    def compute_mask(self, inputs, mask=None):
        # unfortunately this is required
        return 3 * [None]

    def get_config(self):
        configuration = {
            "aspect_ratios": self.aspect_ratios,
            "base_size": self.base_size,
            "clobber_positives": self.clobber_positives,
            "negative_overlap": self.negative_overlap,
            "padding": self.padding,
            "positive_overlap": self.positive_overlap,
            "scales": self.scales,
            "stride": self.stride
        }

        return {**super(Anchor, self).get_config(), **configuration}

    def _balance(self, labels):
        """
        balance labels by setting some to -1
        :param labels: array of labels (1 is positive, 0 is negative, -1 is dont
        care)
        :return: array of labels
        """

        # subsample positive labels if we have too many
        labels = self._subsample_positive_labels(labels)

        # subsample negative labels if we have too many
        labels = self._subsample_negative_labels(labels)

        return labels

    def _label(self, target, output, inds_inside):
        """
        Create bbox labels.
        label: 1 is positive, 0 is negative, -1 is do not care

        :param clobber_positives:
        :param positive_overlap:
        :param negative_overlap:
        :param inds_inside: indices of anchors inside image
        :param output: anchors
        :param target: ground truth objects

        :return: indices of gt boxes with the greatest overlap, balanced labels
        """
        ones = keras.backend.ones_like(inds_inside, dtype=keras.backend.floatx())
        labels = ones * -1
        zeros = keras.backend.zeros_like(inds_inside, dtype=keras.backend.floatx())

        argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = self._overlapping(output, target, inds_inside)

        # Assign background labels first so that positive labels can clobber them.
        if not self.clobber_positives:
            labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, self.negative_overlap), zeros, labels)

        # fg label: for each gt, anchor with highest overlap

        # TODO: generalize unique beyond 1D
        unique_indices, unique_indices_indices = keras_rcnn.backend.unique(gt_argmax_overlaps_inds, return_index=True)

        inverse_labels = keras.backend.gather(-1 * labels, unique_indices)

        unique_indices = keras.backend.expand_dims(unique_indices, 1)

        updates = keras.backend.ones_like(keras.backend.reshape(unique_indices, (-1,)), dtype=keras.backend.floatx())

        labels = keras_rcnn.backend.scatter_add_tensor(labels, unique_indices, inverse_labels + updates)

        # Assign foreground labels based on IoU overlaps that are higher than
        # RPN_POSITIVE_OVERLAP.
        labels = keras_rcnn.backend.where(keras.backend.greater_equal(max_overlaps, self.positive_overlap), ones,
                                          labels)

        if self.clobber_positives:
            # assign bg labels last so that negative labels can clobber positives
            labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, self.negative_overlap), zeros, labels)

        return argmax_overlaps_inds, self._balance(labels)

    @staticmethod
    def _overlapping(output, target, inds_inside):
        """
        overlaps between the anchors and the gt boxes
        :param output: Generated anchors
        :param target: Ground truth bounding boxes
        :param inds_inside:
        :return:
        """

        assert keras.backend.ndim(output) == 2
        assert keras.backend.ndim(target) == 2

        reference = keras_rcnn.backend.intersection_over_union(output, target)

        gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

        argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

        arranged = keras.backend.arange(0, keras.backend.shape(inds_inside)[0])

        indices = keras.backend.stack(
            [arranged, keras.backend.cast(argmax_overlaps_inds, "int32")], axis=0)

        indices = keras.backend.transpose(indices)

        max_overlaps = keras_rcnn.backend.gather_nd(reference, indices)

        return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds

    @staticmethod
    def _subsample_negative_labels(labels, rpn_batchsize=256):
        """
        subsample negative labels if we have too many
        :param labels: array of labels (1 is positive, 0 is negative, -1 is dont
        care)

        :return:
        """
        num_bg = rpn_batchsize - keras.backend.shape(
            keras_rcnn.backend.where(keras.backend.equal(labels, 1)))[0]

        bg_inds = keras_rcnn.backend.where(keras.backend.equal(labels, 0))

        num_bg_inds = keras.backend.shape(bg_inds)[0]

        size = num_bg_inds - num_bg

        def more_negative():
            indices = keras.backend.reshape(bg_inds, (-1,))
            indices = keras_rcnn.backend.shuffle(indices)[:size]

            updates = tensorflow.ones((size,)) * -1

            inverse_labels = keras.backend.gather(labels, indices) * -1

            indices = keras.backend.reshape(indices, (-1, 1))

            return keras_rcnn.backend.scatter_add_tensor(labels, indices, inverse_labels + updates)

        condition = keras.backend.less_equal(size, 0)

        return keras.backend.switch(condition, labels, lambda: more_negative())

    @staticmethod
    def _subsample_positive_labels(labels, rpn_fg_fraction=0.5, rpn_batchsize=256):
        """
        subsample positive labels if we have too many

        :param labels: array of labels (1 is positive, 0 is negative,
        -1 is dont care)

        :return:
        """

        num_fg = int(rpn_fg_fraction * rpn_batchsize)

        fg_inds = keras_rcnn.backend.where(keras.backend.equal(labels, 1))
        num_fg_inds = keras.backend.shape(fg_inds)[0]

        size = num_fg_inds - num_fg

        def more_positive():
            indices = keras.backend.reshape(fg_inds, (-1,))
            indices = keras_rcnn.backend.shuffle(indices)[:size]

            updates = tensorflow.ones((size,)) * -1

            inverse_labels = keras.backend.gather(labels, indices) * -1

            indices = keras.backend.reshape(indices, (-1, 1))

            updates = inverse_labels + updates

            return keras_rcnn.backend.scatter_add_tensor(labels, indices, updates)

        condition = keras.backend.less_equal(size, 0)

        return keras.backend.switch(condition, labels, lambda: more_positive())

    def _unmap(self, data, inds_inside, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """

        if keras.backend.ndim(data) == 1:
            ret = tensorflow.ones((self.k,), dtype=keras.backend.floatx()) * fill

            inds_nd = keras.backend.expand_dims(inds_inside)
        else:
            ret = (self.k, keras.backend.shape(data)[1])
            ret = tensorflow.ones(ret, dtype=keras.backend.floatx()) * fill

            data = keras.backend.transpose(data)
            data = keras.backend.reshape(data, (-1,))

            inds_ii = keras.backend.tile(inds_inside, [4])
            inds_ii = keras.backend.expand_dims(inds_ii)

            ones = keras.backend.expand_dims(keras.backend.ones_like(inds_inside), 1)

            inds_coords = keras.backend.concatenate([ones * 0, ones, ones * 2, ones * 3], 0)

            inds_nd = keras.backend.concatenate([inds_ii, inds_coords], 1)

        inverse_ret = keras_rcnn.backend.gather_nd(-1 * ret, inds_nd)
        inverse_ret = keras_rcnn.backend.squeeze(inverse_ret)

        updates = inverse_ret + data

        ret = keras_rcnn.backend.scatter_add_tensor(ret, inds_nd, updates)

        return ret

    def _inside_image(self, boxes):
        """
        Calc indices of boxes which are located completely inside of the image
        whose size is specified by img_info ((height, width, scale)-shaped array).

        :param boxes: (None, 4) tensor containing boxes in original image
        (x1, y1, x2, y2)

        :param metadata: (height, width, scale)

        :param allowed_border: allow boxes to be outside the image by
        allowed_border pixels

        :return: (None, 4) indices of boxes completely in original image, (None,
        4) tensor of boxes completely inside image
        """

        indices = keras_rcnn.backend.where(
            (boxes[:, 0] >= -self.padding) &
            (boxes[:, 1] >= -self.padding) &
            (boxes[:, 2] < self.padding + self.metadata[0]) &  # width
            (boxes[:, 3] < self.padding + self.metadata[1])  # height
        )

        indices = keras.backend.cast(indices, "int32")

        gathered = keras.backend.gather(boxes, indices)

        return indices[:, 0], keras.backend.reshape(gathered, [-1, 4])

    @staticmethod
    def _inside_and_outside_weights(anchors, subsample, positive_weight, proposed_inside_weights):
        """
        Creates the inside_weights and outside_weights bounding-box weights.

        Args:
            anchors: Generated anchors.
            subsample:  Labels obtained after subsampling.
            positive_weight:
            proposed_inside_weights:

        Returns:
            inside_weights:  Inside bounding-box weights.
            outside_weights: Outside bounding-box weights.
        """
        number_of_anchors = keras.backend.int_shape(anchors)[0]

        proposed_inside_weights = keras.backend.constant([proposed_inside_weights])
        proposed_inside_weights = keras.backend.tile(proposed_inside_weights,
                                                     (number_of_anchors, 1))

        positive_condition = keras.backend.equal(subsample, 1)
        negative_condition = keras.backend.equal(subsample, 0)

        if positive_weight < 0:
            # Assign equal weights to both positive_weights and negative_weights
            # labels.
            examples = keras.backend.cast(negative_condition,
                                          keras.backend.floatx())
            examples = keras.backend.sum(examples)

            positive_weights = keras.backend.ones_like(anchors) / examples
            negative_weights = keras.backend.ones_like(anchors) / examples
        else:
            # Assign weights that favor either the positive or the
            # negative_weights labels.
            assert (positive_weight > 0) & (positive_weight < 1)

            positive_examples = keras.backend.cast(positive_condition,
                                                   keras.backend.floatx())
            positive_examples = keras.backend.sum(positive_examples)

            negative_examples = keras.backend.cast(negative_condition,
                                                   keras.backend.floatx())
            negative_examples = keras.backend.sum(negative_examples)

            positive_weights = keras.backend.ones_like(anchors) * (
                    0 + positive_weight) / positive_examples
            negative_weights = keras.backend.ones_like(anchors) * (
                    1 - positive_weight) / negative_examples

        inside_weights = keras.backend.zeros_like(anchors)
        inside_weights = keras_rcnn.backend.where(positive_condition,
                                                  proposed_inside_weights,
                                                  inside_weights)

        outside_weights = keras.backend.zeros_like(anchors)
        outside_weights = keras_rcnn.backend.where(positive_condition,
                                                   positive_weights,
                                                   outside_weights)
        outside_weights = keras_rcnn.backend.where(negative_condition,
                                                   negative_weights,
                                                   outside_weights)

        return inside_weights, outside_weights