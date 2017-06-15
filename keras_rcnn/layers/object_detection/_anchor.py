import keras.engine
import numpy

import keras_rcnn.backend
import keras_rcnn.losses

RPN_NEGATIVE_OVERLAP = 0.3
RPN_POSITIVE_OVERLAP = 0.7
RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256


class Anchor(keras.engine.topology.Layer):
    """
    Assign anchors to ground-truth targets.

    It produces:
        1. anchor classification labels
        2. bounding-box regression targets.
    """
    def __init__(self, features, image_shape, delta=3, scales=None, ratios=None, stride=16, **kwargs):
        self.delta = delta

        self.features = features

        self.feat_h, self.feat_w = features

        self.image_shape = image_shape

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

        # (feat_h x feat_w x n_anchors, 4)
        self.shifted_anchors = keras_rcnn.backend.shift(self.features, self.stride)

        super(Anchor, self).__init__(**kwargs)

    @property
    def n_anchors(self):
        return len(self.ratios) * len(self.scales)

    def build(self, input_shape):
        super(Anchor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inds_inside, all_inside_bbox = self._inside_image(self.shifted_anchors, self.image_shape)

        argmax_overlaps_inds, bbox_labels = self._label(inputs, all_inside_bbox, inds_inside)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        gt_boxes = inputs[argmax_overlaps_inds]

        bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_bbox, gt_boxes)

        return bbox_labels, bbox_reg_targets, inds_inside, len(self.shifted_anchors)

    def compute_output_shape(self, input_shape):
        return self.output_dim

    @staticmethod
    def _inside_image(y_pred, img_info):
        """
        Calc indicies of anchors which are inside of the image size.
        Calc indicies of anchors which are located completely inside of the image
        whose size is speficied by img_info ((height, width, scale)-shaped array).

        :param y_pred: anchors
        :param img_info:

        :return:
        """
        inds_inside = numpy.where(
            (y_pred[:, 0] >= 0) &
            (y_pred[:, 1] >= 0) &
            (y_pred[:, 2] < img_info[1]) &  # width
            (y_pred[:, 3] < img_info[0])  # height
        )[0]

        return inds_inside, y_pred[inds_inside]

    def _label(self, y_true, y_pred, inds_inside):
        """
        Create bbox labels.
        label: 1 is positive, 0 is negative, -1 is dont care

        :param inds_inside:
        :param y_pred: anchors
        :param y_true:

        :return:
        """
        # assign ignore labels first
        labels = numpy.ones((len(inds_inside),), dtype=numpy.int32) * -1

        argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = self._overlapping(y_true, y_pred, inds_inside)

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps_inds] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1

        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

        labels = self._balance(labels)

        return argmax_overlaps_inds, labels

    def _classification(self):
        pass

    def _regression(self, y_true, y_pred, indicies_inside_image):
        """

        :param y_true:
        :param y_pred:
        :param indicies_inside_image:
        """
        # y_pred has the shape of (1, 4 x n_anchors, feat_h, feat_w)
        # Reshape it into (4, A, K)

        y_true = y_true.reshape(4, self.n_anchors, -1)

        # Transpose it into (K, A, 4)
        y_true = y_true.transpose(2, 1, 0)

        # Reshape it into (K x A, 4)
        y_true = y_true.reshape(-1, 4)

        # Keep the number of bbox
        n_bounding_boxes = y_true.shape[0]

        # Select bbox and ravel it
        y_true = y_true[indicies_inside_image].flatten()

        # Create batch dimension
        y_true = numpy.expand_dims(y_true, 0)

        # Ravel the targets and create batch dimension
        y_pred = y_pred.ravel()[None, :]

        # Calc Smooth L1 Loss (When delta=1, huber loss is SmoothL1Loss)
        # loss = keras_rcnn.losses.logcosh(y_pred, y_true)

        # loss /= n_bounding_boxes
        #
        # return loss.reshape(())

    def _balance(self, labels):
        """

        :param labels:

        :return:
        """
        # subsample positive labels if we have too many
        labels = self._subsample_positive_labels(labels)

        # subsample negative labels if we have too many
        labels = self._subsample_negative_labels(labels)

        return labels

    @staticmethod
    def _subsample_positive_labels(labels):
        """

        :param labels:

        :return:
        """
        num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)

        fg_inds = numpy.where(labels == 1)[0]

        if len(fg_inds) > num_fg:
            size = int(len(fg_inds) - num_fg)

            labels[numpy.random.choice(fg_inds, size, replace=False)] = -1

        return labels

    @staticmethod
    def _subsample_negative_labels(labels):
        """

        :param labels:

        :return:
        """
        num_bg = RPN_BATCHSIZE - numpy.sum(labels[labels == 1])

        bg_inds = numpy.where(labels == 0)[0]

        if len(bg_inds) > num_bg:
            size = bg_inds.shape[0] - num_bg

            labels[numpy.random.choice(bg_inds, size, replace=False)] = -1

        return labels

    @staticmethod
    def _overlapping(y_true, y_pred, inds_inside):
        """
        overlaps between the anchors and the gt boxes

        :param y_pred: anchors
        :param y_true:
        :param inds_inside:

        :return:
        """
        overlaps = keras_rcnn.backend.overlap(y_pred, y_true[:, :4])

        argmax_overlaps_inds = overlaps.argmax(axis=1)
        gt_argmax_overlaps_inds = overlaps.argmax(axis=0)

        max_overlaps = overlaps[numpy.arange(len(inds_inside)), argmax_overlaps_inds]

        gt_max_overlaps = overlaps[gt_argmax_overlaps_inds, numpy.arange(overlaps.shape[1])]

        gt_argmax_overlaps_inds = numpy.where(overlaps == gt_max_overlaps)[0]

        return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds
