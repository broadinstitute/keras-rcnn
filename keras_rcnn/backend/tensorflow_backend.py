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


def crop_and_resize(image, boxes, size):
    """Crop the image given boxes and resize with bilinear interplotation.
    # Parameters
    image: Input image of shape (1, image_height, image_width, depth)
    boxes: Regions of interest of shape (1, num_boxes, 4),
    each row [y1, x1, y2, x2]
    size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    # Returns
    4D Tensor (number of regions, slice_height, slice_width, channels)
    """
    box_ind = tensorflow.zeros_like(boxes, tensorflow.int32)
    box_ind = box_ind[..., 0]
    box_ind = tensorflow.reshape(box_ind, [-1])

    boxes = tensorflow.reshape(boxes, [-1, 4])
    return tensorflow.image.crop_and_resize(image, boxes, box_ind, size)


def overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    overlaps = numpy.zeros((a.shape[0], b.shape[0]), dtype=numpy.float)

    for k, n in itertools.product(range(b.shape[0]), range(a.shape[0])):
        area = ((b[k, 2] - b[k, 0] + 1) * (b[k, 3] - b[k, 1] + 1))

        iw = (min(a[n, 2], b[k, 2]) - max(a[n, 0], b[k, 0]) + 1)

        if iw > 0:
            ih = (min(a[n, 3], b[k, 3]) - max(a[n, 1], b[k, 1]) + 1)

            if ih > 0:
                ua = float((a[n, 2] - a[n, 0] + 1) * (a[n, 3] - a[n, 1] + 1) + area - iw * ih)

                overlaps[n, k] = iw * ih / ua

    return overlaps
