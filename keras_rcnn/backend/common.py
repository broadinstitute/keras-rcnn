import keras.backend
import numpy
import six.moves


def clip(boxes, shape):
    proposals = [
        keras.backend.maximum(keras.backend.minimum(boxes[:, 0::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 1::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 2::4], shape[1] - 1), 0),
        keras.backend.maximum(keras.backend.minimum(boxes[:, 3::4], shape[0] - 1), 0)
    ]

    return keras.backend.concatenate(proposals)


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
