# -*- coding: utf-8 -*-

import tensorflow

import keras_rcnn.backend


def anchor(base_size=16, ratios=None, scales=None):
    """
    Generates a regular grid of multi-aspect and multi-scale anchor boxes.
    """
    if ratios is None:
        ratios = tensorflow.keras.backend.cast(
            [0.5, 1, 2], tensorflow.keras.backend.floatx()
        )

    if scales is None:
        scales = tensorflow.keras.backend.cast(
            [1, 2, 4, 8, 16], tensorflow.keras.backend.floatx()
        )

    base_anchor = tensorflow.keras.backend.cast(
        [-base_size / 2, -base_size / 2, base_size / 2, base_size / 2],
        tensorflow.keras.backend.floatx(),
    )

    base_anchor = tensorflow.keras.backend.expand_dims(base_anchor, 0)

    ratio_anchors = _ratio_enum(base_anchor, ratios)

    anchors = _scale_enum(ratio_anchors, scales)

    anchors = tensorflow.keras.backend.round(anchors)

    return anchors


def bbox_transform(ex_rois, gt_rois):
    """
    Args:
        ex_rois: proposed bounding box coordinates (x1, y1, x2, y2)
        gt_rois: ground truth bounding box coordinates (x1, y1, x2, y2)

    Returns:
        Computed bounding-box regression targets for an image.
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = tensorflow.keras.backend.log(gt_widths / ex_widths)
    targets_dh = tensorflow.keras.backend.log(gt_heights / ex_heights)

    targets = tensorflow.keras.backend.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh)
    )

    targets = tensorflow.keras.backend.transpose(targets)

    return tensorflow.keras.backend.cast(targets, "float32")


def clip(boxes, shape):
    """
    Clips box coordinates to be within the width and height as defined in shape

    """
    indices = tensorflow.keras.backend.tile(
        tensorflow.keras.backend.arange(0, tensorflow.keras.backend.shape(boxes)[0]),
        [4],
    )
    indices = tensorflow.keras.backend.reshape(indices, (-1, 1))
    indices = tensorflow.keras.backend.tile(
        indices, [1, tensorflow.keras.backend.shape(boxes)[1] // 4]
    )
    indices = tensorflow.keras.backend.reshape(indices, (-1, 1))

    indices_coords = tensorflow.keras.backend.tile(
        tensorflow.keras.backend.arange(
            0, tensorflow.keras.backend.shape(boxes)[1], step=4
        ),
        [tensorflow.keras.backend.shape(boxes)[0]],
    )
    indices_coords = tensorflow.keras.backend.concatenate(
        [indices_coords, indices_coords + 1, indices_coords + 2, indices_coords + 3], 0
    )
    indices = tensorflow.keras.backend.concatenate(
        [indices, tensorflow.keras.backend.expand_dims(indices_coords)], axis=1
    )

    updates = tensorflow.keras.backend.concatenate(
        [
            tensorflow.keras.backend.maximum(
                tensorflow.keras.backend.minimum(boxes[:, 0::4], shape[1] - 1), 0
            ),
            tensorflow.keras.backend.maximum(
                tensorflow.keras.backend.minimum(boxes[:, 1::4], shape[0] - 1), 0
            ),
            tensorflow.keras.backend.maximum(
                tensorflow.keras.backend.minimum(boxes[:, 2::4], shape[1] - 1), 0
            ),
            tensorflow.keras.backend.maximum(
                tensorflow.keras.backend.minimum(boxes[:, 3::4], shape[0] - 1), 0
            ),
        ],
        axis=0,
    )
    updates = tensorflow.keras.backend.reshape(updates, (-1,))
    pred_boxes = keras_rcnn.backend.scatter_add_tensor(
        tensorflow.keras.backend.zeros_like(boxes), indices, updates
    )

    return pred_boxes


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    x_ctr = tensorflow.keras.backend.reshape(x_ctr, (-1, 1))
    y_ctr = tensorflow.keras.backend.reshape(y_ctr, (-1, 1))

    col1 = tensorflow.keras.backend.reshape(x_ctr - 0.5 * ws, (-1, 1))
    col2 = tensorflow.keras.backend.reshape(y_ctr - 0.5 * hs, (-1, 1))
    col3 = tensorflow.keras.backend.reshape(x_ctr + 0.5 * ws, (-1, 1))
    col4 = tensorflow.keras.backend.reshape(y_ctr + 0.5 * hs, (-1, 1))
    anchors = tensorflow.keras.backend.concatenate((col1, col2, col3, col4), axis=1)

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size * ratios
    ws = tensorflow.keras.backend.sqrt(size_ratios)
    hs = ws / ratios
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = tensorflow.keras.backend.expand_dims(w, 1) * scales
    hs = tensorflow.keras.backend.expand_dims(h, 1) * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[:, 2] - anchor[:, 0]
    h = anchor[:, 3] - anchor[:, 1]
    x_ctr = anchor[:, 0] + 0.5 * w
    y_ctr = anchor[:, 1] + 0.5 * h
    return w, h, x_ctr, y_ctr


def shift(shape, stride, base_size=16, ratios=None, scales=None):
    """
    Produce shifted anchors based on shape of the map and stride size
    """
    shift_x = tensorflow.keras.backend.arange(0, shape[0] * stride, stride)
    shift_y = tensorflow.keras.backend.arange(0, shape[1] * stride, stride)

    shift_x, shift_y = keras_rcnn.backend.meshgrid(shift_x, shift_y)

    shift_x = tensorflow.keras.backend.reshape(shift_x, [-1])
    shift_y = tensorflow.keras.backend.reshape(shift_y, [-1])

    shifts = tensorflow.keras.backend.stack(
        [shift_x, shift_y, shift_x, shift_y], axis=0
    )

    shifts = tensorflow.keras.backend.transpose(shifts)

    anchors = anchor(base_size=base_size, ratios=ratios, scales=scales)

    number_of_anchors = tensorflow.keras.backend.shape(anchors)[0]

    k = tensorflow.keras.backend.shape(shifts)[
        0
    ]  # number of base points = feat_h * feat_w

    shifted_anchors = tensorflow.keras.backend.reshape(
        anchors, [1, number_of_anchors, 4]
    )

    b = tensorflow.keras.backend.cast(
        tensorflow.keras.backend.reshape(shifts, [k, 1, 4]),
        tensorflow.keras.backend.floatx(),
    )

    shifted_anchors = shifted_anchors + b

    shifted_anchors = tensorflow.keras.backend.reshape(
        shifted_anchors, [k * number_of_anchors, 4]
    )

    return shifted_anchors


def intersection_over_union(output, target):
    """
    Parameters
    ----------
    output: (N, 4) ndarray of float
    target: (K, 4) ndarray of float

    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    intersection_area = (target[:, 2] - target[:, 0] + 1) * (
        target[:, 3] - target[:, 1] + 1
    )

    intersection_c_minimum = tensorflow.keras.backend.minimum(
        tensorflow.keras.backend.expand_dims(output[:, 2], 1), target[:, 2]
    )
    intersection_c_maximum = tensorflow.keras.backend.maximum(
        tensorflow.keras.backend.expand_dims(output[:, 0], 1), target[:, 0]
    )

    intersection_r_minimum = tensorflow.keras.backend.minimum(
        tensorflow.keras.backend.expand_dims(output[:, 3], 1), target[:, 3]
    )
    intersection_r_maximum = tensorflow.keras.backend.maximum(
        tensorflow.keras.backend.expand_dims(output[:, 1], 1), target[:, 1]
    )

    intersection_c = intersection_c_minimum - intersection_c_maximum + 1
    intersection_r = intersection_r_minimum - intersection_r_maximum + 1

    intersection_c = tensorflow.keras.backend.maximum(intersection_c, 0)
    intersection_r = tensorflow.keras.backend.maximum(intersection_r, 0)

    union_area = (
        tensorflow.keras.backend.expand_dims(
            (output[:, 2] - output[:, 0] + 1) * (output[:, 3] - output[:, 1] + 1), 1
        )
        + intersection_area
        - intersection_c * intersection_r
    )

    union_area = tensorflow.keras.backend.maximum(
        union_area, tensorflow.keras.backend.epsilon()
    )

    intersection_area = intersection_c * intersection_r

    return intersection_area / union_area


def smooth_l1(output, target, anchored=False, weights=None):
    difference = tensorflow.keras.backend.abs(output - target)

    p = difference < 1
    q = 0.5 * tensorflow.keras.backend.square(difference)
    r = difference - 0.5

    difference = tensorflow.keras.backend.switch(p, q, r)

    loss = tensorflow.keras.backend.sum(difference, axis=2)

    if weights is not None:
        loss *= weights

    if anchored:
        return loss

    return tensorflow.keras.backend.sum(loss)


def focal_loss(target, output, gamma=2):
    output /= tensorflow.keras.backend.sum(output, axis=-1, keepdims=True)

    epsilon = tensorflow.keras.backend.epsilon()

    output = tensorflow.keras.backend.clip(output, epsilon, 1.0 - epsilon)

    loss = tensorflow.keras.backend.pow(1.0 - output, gamma)

    output = tensorflow.keras.backend.log(output)

    loss = -tensorflow.keras.backend.sum(loss * target * output, axis=-1)

    return loss


def softmax_classification(target, output, anchored=False, weights=None):
    classes = tensorflow.keras.backend.int_shape(output)[-1]

    target = tensorflow.keras.backend.reshape(target, [-1, classes])
    output = tensorflow.keras.backend.reshape(output, [-1, classes])

    loss = tensorflow.keras.backend.categorical_crossentropy(
        target, output, from_logits=False
    )

    if anchored:
        if weights is not None:
            loss = tensorflow.keras.backend.reshape(
                loss, tensorflow.keras.backend.shape(weights)
            )

            loss *= weights

        return loss

    if weights is not None:
        loss *= tensorflow.keras.backend.reshape(weights, [-1])

    return loss


def bbox_transform_inv(boxes, deltas):
    """

    Args:
        boxes: roi or proposal coordinates
        deltas: regression targets

    Returns:
        coordinates as a result of deltas applied to boxes (should be equal to ground truth)
    """

    a = boxes[:, 2] - boxes[:, 0]
    b = boxes[:, 3] - boxes[:, 1]

    ctr_x = boxes[:, 0] + 0.5 * a
    ctr_y = boxes[:, 1] + 0.5 * b

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = (
        dx * a[:, keras_rcnn.backend.newaxis] + ctr_x[:, keras_rcnn.backend.newaxis]
    )
    pred_ctr_y = (
        dy * b[:, keras_rcnn.backend.newaxis] + ctr_y[:, keras_rcnn.backend.newaxis]
    )

    pred_w = tensorflow.keras.backend.exp(dw) * a[:, keras_rcnn.backend.newaxis]
    pred_h = tensorflow.keras.backend.exp(dh) * b[:, keras_rcnn.backend.newaxis]

    indices = tensorflow.keras.backend.tile(
        tensorflow.keras.backend.arange(0, tensorflow.keras.backend.shape(deltas)[0]),
        [4],
    )
    indices = tensorflow.keras.backend.reshape(indices, (-1, 1))
    indices = tensorflow.keras.backend.tile(
        indices, [1, tensorflow.keras.backend.shape(deltas)[-1] // 4]
    )
    indices = tensorflow.keras.backend.reshape(indices, (-1, 1))
    indices_coords = tensorflow.keras.backend.tile(
        tensorflow.keras.backend.arange(
            0, tensorflow.keras.backend.shape(deltas)[1], step=4
        ),
        [tensorflow.keras.backend.shape(deltas)[0]],
    )
    indices_coords = tensorflow.keras.backend.concatenate(
        [indices_coords, indices_coords + 1, indices_coords + 2, indices_coords + 3], 0
    )
    indices = tensorflow.keras.backend.concatenate(
        [indices, tensorflow.keras.backend.expand_dims(indices_coords)], axis=1
    )

    updates = tensorflow.keras.backend.concatenate(
        [
            tensorflow.keras.backend.reshape(pred_ctr_x - 0.5 * pred_w, (-1,)),
            tensorflow.keras.backend.reshape(pred_ctr_y - 0.5 * pred_h, (-1,)),
            tensorflow.keras.backend.reshape(pred_ctr_x + 0.5 * pred_w, (-1,)),
            tensorflow.keras.backend.reshape(pred_ctr_y + 0.5 * pred_h, (-1,)),
        ],
        axis=0,
    )
    pred_boxes = keras_rcnn.backend.scatter_add_tensor(
        tensorflow.keras.backend.zeros_like(deltas), indices, updates
    )

    return pred_boxes
