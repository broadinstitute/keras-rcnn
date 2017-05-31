import itertools

import keras.backend
import numpy
import tensorflow

import keras_rcnn.backend


def bbox_transform_inv(shifted, boxes):
    if boxes.shape[0] == 0:
        return tensorflow.zeros((0, boxes.shape[1]), dtype=boxes.dtype)

    a = shifted[:, 2] - shifted[:, 0] + 1.0
    b = shifted[:, 3] - shifted[:, 1] + 1.0

    ctr_x = shifted[:, 0] + 0.5 * a
    ctr_y = shifted[:, 1] + 0.5 * b

    dx = boxes[:, 0::4]
    dy = boxes[:, 1::4]
    dw = boxes[:, 2::4]
    dh = boxes[:, 3::4]

    pred_ctr_x = dx * a[:, tensorflow.newaxis] + ctr_x[:, tensorflow.newaxis]
    pred_ctr_y = dy * b[:, tensorflow.newaxis] + ctr_y[:, tensorflow.newaxis]

    pred_w = tensorflow.exp(dw) * a[:, tensorflow.newaxis]
    pred_h = tensorflow.exp(dh) * b[:, tensorflow.newaxis]

    pred_boxes = [
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ]

    return keras.backend.concatenate(pred_boxes)


def filter_boxes(proposals, minimum):
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indicies = tensorflow.where((ws >= minimum) & (hs >= minimum))

    indicies = keras.backend.flatten(indicies)

    return keras.backend.cast(indicies, tensorflow.int32)


def non_maximum_suppression(boxes, scores, maximum, threshold=0.5):
    return tensorflow.image.non_max_suppression(
        boxes=boxes,
        iou_threshold=threshold,
        max_output_size=maximum,
        scores=scores
    )


def propose(boxes, scores, maximum):
    shape = keras.backend.int_shape(boxes)[1:3]

    shifted = keras_rcnn.backend.shift(shape, 16)

    proposals = keras.backend.reshape(boxes, (-1, 4))

    proposals = keras_rcnn.backend.bbox_transform_inv(shifted, proposals)

    proposals = keras_rcnn.backend.clip(proposals, shape)

    indicies = keras_rcnn.backend.filter_boxes(proposals, 1)

    proposals = keras.backend.gather(proposals, indicies)

    scores = scores[:, :, :, :9]
    scores = keras.backend.reshape(scores, (-1, 1))
    scores = keras.backend.gather(scores, indicies)
    scores = keras.backend.flatten(scores)

    proposals = keras.backend.cast(proposals, tensorflow.float32)
    scores = keras.backend.cast(scores, tensorflow.float32)

    indicies = keras_rcnn.backend.non_maximum_suppression(proposals, scores, maximum, 0.7)

    proposals = keras.backend.gather(proposals, indicies)

    return keras.backend.expand_dims(proposals, 0)


def resize_images(images, shape):
    return tensorflow.image.resize_images(images, shape)


def overlap(x, y):
    n = x.shape[0]

    k = y.shape[0]

    overlaps = keras.backend.zeros((n, k), dtype=numpy.float32)

    for k_index in range(k):
        area = ((y[k_index, 2] - y[k_index, 0] + 1) * (y[k_index, 3] - y[k_index, 1] + 1))

        for n_index in range(n):
            iw = (min(x[n_index, 2], y[k_index, 2]) - max(x[n_index, 0], y[k_index, 0]) + 1)

            if iw > 0:
                ih = (min(x[n_index, 3], y[k_index, 3]) - max(x[n_index, 1], y[k_index, 1]) + 1)

                if ih > 0:
                    ua = float((x[n_index, 2] - x[n_index, 0] + 1) * (x[n_index, 3] - x[n_index, 1] + 1) + area - iw * ih)

                    overlaps[n, k] = iw * ih / ua

    return overlaps


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = numpy.zeros((N, K), dtype=numpy.float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))

        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)

            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)

                    overlaps[n, k] = iw * ih / ua

    return overlaps
