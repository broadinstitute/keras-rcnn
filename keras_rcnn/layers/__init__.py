import keras.engine.topology
import numpy
import six.moves
import tensorflow

from .pooling import ROI


class Proposal(keras.engine.topology.Layer):
    def __init__(self, proposals, **kwargs):
        self.output_dim = (None, None, 4)

        self.proposals = proposals

        super(Proposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Proposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return propose(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return self.output_dim


def anchor(base_size=15, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    base_anchor = numpy.array([0, 0, base_size, base_size])

    ratio_anchors = _ratio_enum(base_anchor, numpy.asarray(ratios))

    anchors = numpy.vstack(
        [_scale_enum(ratio_anchors[i, :], numpy.asarray(scales))
         for i in six.moves.range(len(ratio_anchors))])

    return anchors


def _whctrs(anchor):
    # Return width, height, x center, and y center for an anchor (window).
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    # Given a vector of widths (ws) and heights (hs) around a center
    # (x_ctr, y_ctr), output a set of anchors (windows).
    ws, hs = ws[:, None], hs[:, None]
    anchors = numpy.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    # Enumerate a set of anchors for each aspect ratio wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = numpy.rint(numpy.sqrt(size_ratios))
    hs = numpy.rint(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    # Enumerate a set of anchors for each scale wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


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


def clip(boxes, shape):
    proposals = [
        keras.backend.maximum(keras.backend.minimum(boxes[:, 0::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 1::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 2::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 3::4], shape[0] - 1), 0)
    ]

    return keras.backend.concatenate(proposals)


def filter_boxes(proposals, minimum):
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indicies = tensorflow.where((ws >= minimum) & (hs >= minimum))

    indicies = keras.backend.flatten(indicies)

    return keras.backend.cast(indicies, tensorflow.int32)


def shift(shape, stride):
    shift_x = numpy.arange(0, shape[0]) * stride
    shift_y = numpy.arange(0, shape[1]) * stride

    shift_x, shift_y = numpy.meshgrid(shift_x, shift_y)

    shifts = numpy.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    anchors = anchor()

    # Create all bbox
    number_of_anchors = len(anchors)

    k = len(shifts)  # number of base points = feat_h * feat_w

    bbox = anchors.reshape(1, number_of_anchors, 4) + shifts.reshape(k, 1, 4)

    bbox = bbox.reshape(k * number_of_anchors, 4)

    return bbox


def propose(boxes, scores):
    shape = keras.backend.int_shape(boxes)[1:3]

    shifted = shift(shape, 16)

    proposals = keras.backend.reshape(boxes, (-1, 4))

    proposals = bbox_transform_inv(shifted, proposals)

    proposals = clip(proposals, shape)

    indicies = filter_boxes(proposals, 1)

    proposals = keras.backend.gather(proposals, indicies)

    scores = scores[:, :, :, :9]
    scores = keras.backend.reshape(scores, (-1, 1))
    scores = keras.backend.gather(scores, indicies)
    scores = keras.backend.flatten(scores)

    proposals = keras.backend.cast(proposals, tensorflow.float32)
    scores = keras.backend.cast(scores, tensorflow.float32)

    indicies = tensorflow.image.non_max_suppression(proposals, scores, 100, 0.7)

    proposals = keras.backend.gather(proposals, indicies)

    return keras.backend.expand_dims(proposals, 0)
