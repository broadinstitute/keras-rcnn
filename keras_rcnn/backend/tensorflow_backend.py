import keras.backend
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


def propose(boxes, scores):
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

    indicies = keras_rcnn.backend.non_maximum_suppression(proposals, scores, 100, 0.7)

    proposals = keras.backend.gather(proposals, indicies)

    return keras.backend.expand_dims(proposals, 0)


def resize_images(images, shape):
    return tensorflow.image.resize_images(images, shape)
