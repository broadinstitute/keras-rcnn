import keras.backend
import numpy
import numpy.testing

import keras_rcnn.backend
import keras_rcnn.backend.common
import keras_rcnn.layers.object_detection._anchor_target
import keras_rcnn.layers.object_detection._object_proposal


def test_anchor():
    x = numpy.array(
        [[-84., -40., 99., 55.],
         [-176., -88., 191., 103.],
         [-360., -184., 375., 199.],
         [-56., -56., 71., 71.],
         [-120., -120., 135., 135.],
         [-248., -248., 263., 263.],
         [-36., -80., 51., 95.],
         [-80., -168., 95., 183.],
         [-168., -344., 183., 359.]]
    )

    y = keras_rcnn.backend.anchor(
        scales=keras.backend.cast([8, 16, 32], keras.backend.floatx()))
    y = keras.backend.eval(y)
    numpy.testing.assert_array_almost_equal(x, y)


def test_clip():
    boxes = numpy.array(
        [[0, 0, 0, 0], [1, 2, 3, 4], [-4, 2, 1000, 6000], [3, -10, 223, 224]])
    shape = [224, 224]
    boxes = keras.backend.variable(boxes)
    results = keras_rcnn.backend.clip(boxes, shape)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[0, 0, 0, 0], [1, 2, 3, 4], [0, 2, 223, 223], [3, 0, 223, 223]])
    numpy.testing.assert_array_almost_equal(results, expected)

    boxes = numpy.reshape(numpy.arange(200, 200 + 12 * 5), (-1, 12))
    shape = [224, 224]
    boxes = keras.backend.variable(boxes)
    results = keras_rcnn.backend.clip(boxes, shape)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211],
         [212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223],
         [223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223],
         [223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223],
         [223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223]])
    numpy.testing.assert_array_almost_equal(results, expected, 0)


def test_bbox_transform():
    gt_rois = numpy.array([[-84., -40., 99., 55.], [-176., -88., 191., 103.],
                           [-360., -184., 375., 199.], [-56., -56., 71., 71.],
                           [-120., -120., 135., 135.],
                           [-248., -248., 263., 263.], [-36., -80., 51., 95.],
                           [-80., -168., 95., 183.],
                           [-168., -344., 183., 359.]])
    ex_rois = 2 * gt_rois
    gt_rois = keras.backend.variable(gt_rois)
    ex_rois = keras.backend.variable(ex_rois)
    results = keras_rcnn.backend.bbox_transform(ex_rois, gt_rois)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[-0.02043597, -0.03926702, -0.69042609, -0.68792524],
         [-0.01020408, -0.01958225, -0.69178756, -0.69053962],
         [-0.00509857, -0.00977836, -0.6924676, -0.69184425],
         [-0.02941176, -0.02941176, -0.68923328, -0.68923328],
         [-0.0146771, -0.0146771, -0.69119215, -0.69119215],
         [-0.00733138, -0.00733138, -0.69217014, -0.69217014],
         [-0.04285714, -0.02136752, -0.68744916, -0.69030223],
         [-0.02136752, -0.01066856, -0.69030223, -0.69172572],
         [-0.01066856, -0.00533049, -0.69172572, -0.6924367]])
    numpy.testing.assert_array_almost_equal(results, expected)


def test_mkanchors():
    ws = numpy.array([1, 2, 3])
    hs = numpy.array([4, 5, 6])
    x_ctr = keras.backend.variable([1], 'float32')
    y_ctr = keras.backend.variable([2], 'float32')
    ws = keras.backend.variable(ws, 'float32')
    hs = keras.backend.variable(hs, 'float32')
    results = keras_rcnn.backend.common._mkanchors(ws, hs, x_ctr, y_ctr)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[1, 0.5, 1, 3.5], [0.5, 0, 1.5, 4], [0, -0.5, 2, 4.5]])
    numpy.testing.assert_array_equal(results, expected)


def test_overlap():
    x = numpy.asarray([
        [0, 10, 0, 10],
        [0, 20, 0, 20],
        [0, 30, 0, 30],
        [0, 40, 0, 40],
        [0, 50, 0, 50],
        [0, 60, 0, 60],
        [0, 70, 0, 70],
        [0, 80, 0, 80],
        [0, 90, 0, 90]
    ])
    x = keras.backend.variable(x)

    y = numpy.asarray([
        [0, 20, 0, 20],
        [0, 40, 0, 40],
        [0, 60, 0, 60],
        [0, 80, 0, 80]
    ])
    y = keras.backend.variable(y)

    overlapping = keras_rcnn.backend.common.overlap(x, y)

    overlapping = keras.backend.eval(overlapping)

    expected = numpy.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    numpy.testing.assert_array_equal(overlapping, expected)


def test_ratio_enum():
    anchor = numpy.expand_dims(numpy.array([0, 0, 0, 0]), 0)
    ratios = numpy.array([1, 2, 3])
    anchor = keras.backend.variable(anchor)
    ratios = keras.backend.variable(ratios)
    results = keras_rcnn.backend.common._ratio_enum(anchor, ratios)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[0., 0., 0., 0.], [0., -0.5, 0., 0.5], [0., -1., 0., 1.]])
    numpy.testing.assert_array_equal(results, expected)
    anchor = numpy.expand_dims(numpy.array([2, 3, 100, 100]), 0)
    anchor = keras.backend.variable(anchor)
    results = keras_rcnn.backend.common._ratio_enum(anchor, ratios)
    results = keras.backend.eval(results)
    expected = numpy.array([[2.5, 3., 99.5, 100.], [16.5, -18., 85.5, 121.],
                            [23., -33.5, 79., 136.5]])
    numpy.testing.assert_array_equal(results, expected)


def test_scale_enum():
    anchor = numpy.expand_dims(numpy.array([0, 0, 0, 0]), 0)
    scales = numpy.array([1, 2, 3])
    anchor = keras.backend.variable(anchor)
    scales = keras.backend.variable(scales)
    results = keras_rcnn.backend.common._scale_enum(anchor, scales)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[0, 0, 0, 0], [-0.5, -0.5, 0.5, 0.5], [-1., -1., 1., 1.]])
    numpy.testing.assert_array_equal(results, expected)
    anchor = keras.backend.cast(
        numpy.expand_dims(numpy.array([2, 3, 100, 100]), 0), 'float32')
    anchor = keras.backend.variable(anchor)
    results = keras_rcnn.backend.common._scale_enum(anchor, scales)
    results = keras.backend.eval(results)
    expected = numpy.array([[2., 3., 100., 100.], [-47.5, -46., 149.5, 149.],
                            [-97., -95., 199., 198.]])
    numpy.testing.assert_array_equal(results, expected)


def test_whctrs():
    anchor = keras.backend.cast(keras.backend.expand_dims([0, 0, 0, 0], 0),
                                'float32')
    results0, results1, results2, results3 = keras_rcnn.backend.common._whctrs(
        anchor)
    results = numpy.array(
        [keras.backend.eval(results0), keras.backend.eval(results1),
         keras.backend.eval(results2), keras.backend.eval(results3)])
    expected = numpy.expand_dims([1, 1, 0, 0], 1)
    numpy.testing.assert_array_equal(results, expected)
    anchor = keras.backend.cast(keras.backend.expand_dims([2, 3, 100, 100], 0),
                                'float32')
    results0, results1, results2, results3 = keras_rcnn.backend.common._whctrs(
        anchor)
    results = numpy.array(
        [keras.backend.eval(results0), keras.backend.eval(results1),
         keras.backend.eval(results2), keras.backend.eval(results3)])
    expected = numpy.expand_dims([99, 98, 51, 51.5], 1)
    numpy.testing.assert_array_equal(results, expected)


def test_shift():
    y = keras_rcnn.backend.shift((14, 14), 16)

    assert keras.backend.int_shape(y) == (1764, 4)

    assert y.dtype == keras.backend.floatx()


def test_bbox_transform_inv():
    anchors = 9
    features = (14, 14)
    shifted = keras_rcnn.backend.shift(features, 16)
    deltas = numpy.zeros((features[0] * features[1] * anchors, 4))
    deltas = keras.backend.variable(deltas)
    pred_boxes = keras_rcnn.backend.bbox_transform_inv(shifted, deltas)
    assert keras.backend.eval(pred_boxes).shape == (1764, 4)

    shifted = numpy.zeros((5, 4))
    deltas = numpy.reshape(numpy.arange(12 * 5), (5, -1))
    deltas = keras.backend.variable(deltas)
    pred_boxes = keras_rcnn.backend.bbox_transform_inv(shifted, deltas)
    expected = numpy.array(
        [[-3.19452805e+00, -8.54276846e+00, 4.19452805e+00,
          1.15427685e+01, -1.97214397e+02, -5.42816579e+02,
          2.06214397e+02, 5.53816579e+02, -1.10047329e+04,
          -2.99275709e+04, 1.10217329e+04, 2.99465709e+04],
         [-6.01289642e+05, -1.63449519e+06, 6.01314642e+05,
          1.63452219e+06, -3.28299681e+07, -8.92411330e+07,
          3.28300011e+07, 8.92411680e+07, -1.79245640e+09,
          -4.87240170e+09, 1.79245644e+09, 4.87240174e+09],
         [-9.78648047e+10, -2.66024120e+11, 9.78648047e+10,
          2.66024120e+11, -5.34323729e+12, -1.45244248e+13,
          5.34323729e+12, 1.45244248e+13, -2.91730871e+14,
          -7.93006726e+14, 2.91730871e+14, 7.93006726e+14],
         [-1.59279659e+16, -4.32967002e+16, 1.59279659e+16,
          4.32967002e+16, -8.69637471e+17, -2.36391973e+18,
          8.69637471e+17, 2.36391973e+18, -4.74805971e+19,
          -1.29065644e+20, 4.74805971e+19, 1.29065644e+20],
         [-2.59235276e+21, -7.04674541e+21, 2.59235276e+21,
          7.04674541e+21, -1.41537665e+23, -3.84739263e+23,
          1.41537665e+23, 3.84739263e+23, -7.72769468e+24,
          -2.10060520e+25, 7.72769468e+24, 2.10060520e+25]],
        dtype=numpy.float32)
    numpy.testing.assert_array_almost_equal(keras.backend.eval(pred_boxes)[0],
                                            expected[0], 0, verbose=True)


def test_smooth_l1():
    output = keras.backend.variable(
        [[[2.5, 0.0, 0.4, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 2.5, 0.0, 0.4]],
         [[3.5, 0.0, 0.0, 0.0],
          [0.0, 0.4, 0.0, 0.9],
          [0.0, 0.0, 1.5, 0.0]]]
    )

    target = keras.backend.zeros_like(output)

    x = keras_rcnn.backend.smooth_l1(output, target)

    numpy.testing.assert_approx_equal(keras.backend.eval(x), 8.645)

    weights = keras.backend.variable(
        [[2, 1, 1],
         [0, 3, 0]]
    )

    x = keras_rcnn.backend.smooth_l1(output, target, weights=weights)

    numpy.testing.assert_approx_equal(keras.backend.eval(x), 7.695)


def test_softmax_classification():
    output = [[[-100, 100, -100],
              [100, -100, -100],
              [0, 0, -100],
              [-100, -100, 100]],
              [[-100, 0, 0],
              [-100, 100, -100],
              [-100, 100, -100],
              [100, -100, -100]]]

    target = [[[0, 1, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 0, 1]],
              [[0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0]]]

    weights = [[1.0, 1.0, 0.5, 1.0],
               [1.0, 1.0, 1.0, 0.0]]

    x = keras_rcnn.backend.softmax_classification(
        keras.backend.variable(output),
        keras.backend.variable(target),
        weights=keras.backend.variable(weights)
    )

    output_y = numpy.reshape(output, [-1, 3])
    target_y = numpy.reshape(target, [-1, 3])
    output_y = output_y / numpy.sum(output_y, axis=1, keepdims=True)

    # epsilon = keras.backend.epsilon()
    # stop python for complaining weakref
    epsilon = 1e-07

    output_y = numpy.clip(output_y,
                          epsilon,
                          1. - epsilon)
    _y = - numpy.sum(target_y * numpy.log(output_y), axis=1)
    y = _y * numpy.reshape(weights, -1)

    numpy.testing.assert_array_almost_equal(keras.backend.eval(x), y)

    x = keras_rcnn.backend.softmax_classification(
        keras.backend.variable(output),
        keras.backend.variable(target),
        weights=keras.backend.variable(weights),
        anchored=True
    )

    y = weights * numpy.reshape(_y, numpy.asarray(weights).shape)
    numpy.testing.assert_array_almost_equal(keras.backend.eval(x), y)
