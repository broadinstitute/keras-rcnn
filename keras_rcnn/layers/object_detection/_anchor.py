import keras.engine
import numpy

import keras_rcnn.backend


class Anchor(keras.engine.topology.Layer):
    """
    Assign anchors to ground-truth targets.

    It produces:
        1. anchor classification labels
        2. bounding-box regression targets.
    """
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 256

    def __init__(self, features, img_info, stride=16, scales=None, ratios=None, **kwargs):
        self.feat_h, self.feat_w = features

        self.img_info = img_info

        self.output_dim = (None, None, 4)

        if ratios is None:
            self.ratios = [0.5, 1, 2]
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = [8, 16, 32]
        else:
            self.scales = scales

        self.stride = stride

        super(Anchor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Anchor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gt_boxes = inputs

        return self._call(gt_boxes)

    def compute_output_shape(self, input_shape):
        return self.output_dim

    def _call(self, inputs):
        # (feat_h x feat_w x n_anchors, 4)
        all_bbox = self._generate_all_bbox_use_array_info(self.feat_h, self.feat_w)

        inds_inside, all_inside_bbox = self.inside_image(all_bbox, self.img_info)

        argmax_overlaps_inds, bbox_labels = self.label(inds_inside, all_inside_bbox, inputs)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        gt_boxes = inputs[argmax_overlaps_inds]
        bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_bbox, gt_boxes)

        return bbox_labels, bbox_reg_targets, inds_inside, len(all_bbox)

    def _generate_all_bbox_use_array_info(self, feat_h, feat_w):
        return keras_rcnn.backend.shift((feat_h, feat_w), self.stride)

    @staticmethod
    def inside_image(anchors, img_info):
        """
        Calc indicies of anchors which are inside of the image size.
        Calc indicies of anchors which are located completely inside of the image
        whose size is speficied by img_info ((height, width, scale)-shaped array).

        :param anchors:
        :param img_info:

        :return:
        """
        inds_inside = numpy.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] < img_info[1]) &  # width
            (anchors[:, 3] < img_info[0])  # height
        )[0]

        return inds_inside, anchors[inds_inside]

    def label(self, inds_inside, anchors, gt_boxes):
        """
        Create bbox labels.
        label: 1 is positive, 0 is negative, -1 is dont care

        :param inds_inside:
        :param anchors:
        :param gt_boxes:

        :return:
        """
        # assign ignore labels first
        labels = numpy.ones((len(inds_inside),), dtype=numpy.int32) * -1

        argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = self.overlapping(anchors, gt_boxes, inds_inside)

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps_inds] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        labels = self.subsample_positive_labels(labels)

        # subsample negative labels if we have too many
        labels = self.subsample_negative_labels(labels)

        return argmax_overlaps_inds, labels

    def subsample_positive_labels(self, labels):
        """

        :param labels:

        :return:
        """
        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)

        fg_inds = numpy.where(labels == 1)[0]

        if len(fg_inds) > num_fg:
            disable_inds = numpy.random.choice(fg_inds, size=int(len(fg_inds) - num_fg), replace=False)

            labels[disable_inds] = -1

        return labels

    def subsample_negative_labels(self, labels):
        """

        :param labels:

        :return:
        """
        num_bg = self.RPN_BATCHSIZE - numpy.sum(labels == 1)

        bg_inds = numpy.where(labels == 0)[0]

        if len(bg_inds) > num_bg:
            disable_inds = numpy.random.choice(bg_inds, size=int(len(bg_inds) - num_bg), replace=False)

            labels[disable_inds] = -1

        return labels

    @staticmethod
    def overlapping(anchors, gt_boxes, inds_inside):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        overlaps = keras_rcnn.backend.bbox_overlaps(anchors, gt_boxes[:, :4])

        # TODO(mitmul): Remove this when bbox_overlaps for GPU comes
        overlaps = numpy.asarray(overlaps)

        argmax_overlaps_inds = overlaps.argmax(axis=1)
        gt_argmax_overlaps_inds = overlaps.argmax(axis=0)

        max_overlaps = overlaps[numpy.arange(len(inds_inside)), argmax_overlaps_inds]

        gt_max_overlaps = overlaps[gt_argmax_overlaps_inds, numpy.arange(overlaps.shape[1])]

        gt_argmax_overlaps_inds = numpy.where(overlaps == gt_max_overlaps)[0]

        return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds