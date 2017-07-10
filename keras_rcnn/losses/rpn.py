import keras.backend
import keras.losses
import tensorflow

import keras_rcnn.backend


def separate_pred(y_pred):
    anchors = keras.backend.shape(y_pred)[-1] // 5
    return y_pred[:, :, :, 0: 4 * anchors], y_pred[:, :, :, 4 * anchors: 5 * anchors]


def encode(features, image_shape, y_true, stride=16):
    # get all anchors inside bbox
    shifted_anchors = keras_rcnn.backend.shift(features, stride)
    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(shifted_anchors, image_shape)

    # indices of gt boxes with the greatest overlap, bbox labels
    # TODO: assert y_true.shape[0] == 1
    y_true = y_true[0, :, :]
    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(y_true, all_inside_bbox, inds_inside)

    # gt boxes
    gt_boxes = keras.backend.gather(y_true, argmax_overlaps_inds)

    # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_bbox, gt_boxes)

    return bbox_labels, bbox_reg_targets, inds_inside


def proposal(anchors, image_shape, stride, *args, **kwargs):
    def f(y_true, y_pred):
        # separate y_pred into rpn_cls_pred and rpn_reg_pred
        y_pred_regression, y_pred_classification = separate_pred(y_pred)

        # convert y_true from gt_boxes to gt_anchors
        features = y_pred.get_shape().as_list()[1:3]

        gt_classification, gt_regression, inds_inside = encode(features, image_shape, y_true, stride)

        y_true_classification = keras.backend.zeros((1, features[0], features[1], anchors * 2))

        y_true_regression = keras.backend.zeros((1, features[0], features[1], anchors * 8))

        z = keras.backend.expand_dims(keras.backend.zeros_like(inds_inside), 1)

        i = keras.backend.expand_dims(inds_inside // (anchors * features[1]), 1)

        j = keras.backend.expand_dims(inds_inside // anchors % features[1], 1)

        a = keras.backend.expand_dims((inds_inside % anchors), 1)

        zz = keras.backend.tile(z, [4, 1])

        ii = keras.backend.tile(i, [4, 1])

        jj = keras.backend.tile(j, [4, 1])

        aa = keras.backend.concatenate([4 * a, 4 * a + 1, 4 * a + 2, 4 * a + 3], 0)

        gt_r1 = keras.backend.tile(gt_classification, [4])

        gt_r1_mask = keras.backend.equal(keras.backend.abs(gt_r1), 1.0)

        gt_r1_mask = keras.backend.cast(gt_r1_mask, keras.backend.floatx())

        gt_c1_mask = keras.backend.equal(keras.backend.abs(gt_classification), 1.0)

        gt_c1_mask = keras.backend.cast(gt_c1_mask, keras.backend.floatx())

        gt_c2_mask = keras.backend.equal(keras.backend.abs(gt_classification), 1.0)

        gt_c2_mask = keras.backend.cast(gt_c2_mask, keras.backend.floatx())

        y_true_classification_indices = keras.backend.concatenate([z, i, j, a], 1)

        y_true_classification = keras_rcnn.backend.scatter_add_tensor(y_true_classification, y_true_classification_indices, gt_c1_mask)

        y_true_classification_indices = keras.backend.concatenate([z, i, j, anchors + a], 1)

        y_true_classification = keras_rcnn.backend.scatter_add_tensor(y_true_classification, y_true_classification_indices, gt_c2_mask)

        y_true_regression_indices = keras.backend.concatenate([zz, ii, jj, aa], 1)

        y_true_regression = keras_rcnn.backend.scatter_add_tensor(y_true_regression, y_true_regression_indices, gt_r1_mask)

        y_true_regression_indices = keras.backend.concatenate([zz, ii, jj, anchors * 4 + aa], 1)

        y_true_regression_updates = keras.backend.reshape(gt_regression, (-1,))

        y_true_regression = keras_rcnn.backend.scatter_add_tensor(y_true_regression, y_true_regression_indices, y_true_regression_updates)

        y_true_classification = keras.backend.reshape(y_true_classification, (1, features[0], features[1], anchors * 2))

        y_true_regression = keras.backend.reshape(y_true_regression, (1, features[0], features[1], anchors * 8))

        classification = _classification(anchors=anchors)(y_true_classification, y_pred_classification)

        regression = _regression(anchors=anchors)(y_true_regression, y_pred_regression)

        return classification, regression

    return f


def _classification(anchors=9):
    """
    Return the classification loss of region proposal network.

    :param anchors: Integer, number of anchors at each sliding position. Equal to number of scales * number of aspect ratios.

    :return: A loss function for region propose classification.
    """

    def f(y_true, y_pred):
        # Binary classification loss
        x, y = y_pred[:, :, :, :], y_true[:, :, :, anchors:]

        a = y_true[:, :, :, :anchors] * keras.backend.binary_crossentropy(x, y)
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + y_true[:, :, :, :anchors]
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f


def _regression(anchors=9):
    """
    Return the regression loss of region proposal network.

    :param anchors: Integer, number of anchors at each sliding position. Equal to number of scales * number of aspect ratios.

    :return: A loss function region propose regression.
    """

    def f(y_true, y_pred):
        # Robust L1 Loss
        x = y_true[:, :, :, 4 * anchors:] - y_pred

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = y_true[:, :, :, :4 * anchors]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f
