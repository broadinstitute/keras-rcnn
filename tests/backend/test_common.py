import keras.backend
import numpy
import numpy.testing

import keras_rcnn.backend
import keras_rcnn.backend.common
import keras_rcnn.layers.object_detection._anchor
import keras_rcnn.layers.object_detection._object_proposal


def test_anchor():
    x = numpy.array(
        [
            [-8.0, -8.0, 8.0, 8.0],
            [-16.0, -16.0, 16.0, 16.0],
            [-24.0, -24.0, 24.0, 24.0],
            [-11.0, -6.0, 11.0, 6.0],
            [-23.0, -11.0, 23.0, 11.0],
            [-34.0, -17.0, 34.0, 17.0],
            [-14.0, -5.0, 14.0, 5.0],
            [-28.0, -9.0, 28.0, 9.0],
            [-42.0, -14.0, 42.0, 14.0],
        ]
    )

    y = keras_rcnn.backend.anchor(
        ratios=keras.backend.cast([1, 2, 3], keras.backend.floatx()),
        scales=keras.backend.cast([1, 2, 3], keras.backend.floatx()),
    )
    y = keras.backend.eval(y)

    numpy.testing.assert_array_almost_equal(x, y, 0)


def test_clip():
    boxes = numpy.array(
        [[0, 0, 0, 0], [1, 2, 3, 4], [-4, 2, 1000, 6000], [3, -10, 223, 224]]
    )
    shape = [224, 224]
    boxes = keras.backend.variable(boxes)
    results = keras_rcnn.backend.clip(boxes, shape)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[0, 0, 0, 0], [1, 2, 3, 4], [0, 2, 223, 223], [3, 0, 223, 223]]
    )
    numpy.testing.assert_array_almost_equal(results, expected)

    boxes = numpy.reshape(numpy.arange(200, 200 + 12 * 5), (-1, 12))
    shape = [224, 224]
    boxes = keras.backend.variable(boxes)
    results = keras_rcnn.backend.clip(boxes, shape)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [
            [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211],
            [212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223],
            [223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223],
            [223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223],
            [223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223],
        ]
    )
    numpy.testing.assert_array_almost_equal(results, expected, 0)


def test_bbox_transform():
    gt_rois = numpy.array(
        [
            [-84.0, -40.0, 99.0, 55.0],
            [-176.0, -88.0, 191.0, 103.0],
            [-360.0, -184.0, 375.0, 199.0],
            [-56.0, -56.0, 71.0, 71.0],
            [-120.0, -120.0, 135.0, 135.0],
            [-248.0, -248.0, 263.0, 263.0],
            [-36.0, -80.0, 51.0, 95.0],
            [-80.0, -168.0, 95.0, 183.0],
            [-168.0, -344.0, 183.0, 359.0],
        ]
    )
    ex_rois = 2 * gt_rois
    results = keras_rcnn.backend.bbox_transform(ex_rois, gt_rois)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [
            [
                -0.020491803278688523,
                -0.039473684210526314,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.010217983651226158,
                -0.01963350785340314,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.00510204081632653,
                -0.0097911227154047,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.02952755905511811,
                -0.02952755905511811,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.014705882352941176,
                -0.014705882352941176,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.007338551859099804,
                -0.007338551859099804,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.04310344827586207,
                -0.02142857142857143,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.02142857142857143,
                -0.010683760683760684,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
            [
                -0.010683760683760684,
                -0.005334281650071123,
                -0.6931471805599453,
                -0.6931471805599453,
            ],
        ]
    )
    numpy.testing.assert_array_almost_equal(results, expected)


def test_mkanchors():
    ws = numpy.array([1, 2, 3])
    hs = numpy.array([4, 5, 6])
    x_ctr = keras.backend.variable([0], "float32")
    y_ctr = keras.backend.variable([2], "float32")
    ws = keras.backend.variable(ws, "float32")
    hs = keras.backend.variable(hs, "float32")
    results = keras_rcnn.backend.common._mkanchors(ws, hs, x_ctr, y_ctr)
    results = keras.backend.eval(results)
    expected = numpy.array(
        [[-0.5, 0.0, 0.5, 4.0], [-1.0, -0.5, 1.0, 4.5], [-1.5, -1.0, 1.5, 5.0]]
    )
    numpy.testing.assert_array_equal(results, expected)


def test_overlap():
    x = numpy.asarray(
        [
            [0, 10, 0, 10],
            [0, 20, 0, 20],
            [0, 30, 0, 30],
            [0, 40, 0, 40],
            [0, 50, 0, 50],
            [0, 60, 0, 60],
            [0, 70, 0, 70],
            [0, 80, 0, 80],
            [0, 90, 0, 90],
        ]
    )
    x = keras.backend.variable(x)

    y = numpy.asarray([[0, 20, 0, 20], [0, 40, 0, 40], [0, 60, 0, 60], [0, 80, 0, 80]])
    y = keras.backend.variable(y)

    overlapping = keras_rcnn.backend.common.intersection_over_union(x, y)

    overlapping = keras.backend.eval(overlapping)

    expected = numpy.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    numpy.testing.assert_array_equal(overlapping, expected)


def test_ratio_enum():
    anchor = numpy.expand_dims(numpy.array([0, 0, 16, 16]), 0)
    ratios = numpy.array([0.5, 2])
    anchor = keras.backend.variable(anchor)
    ratios = keras.backend.variable(ratios)
    results = keras_rcnn.backend.common._ratio_enum(anchor, ratios)
    results = keras.backend.round(results)
    results = keras.backend.eval(results)
    expected = numpy.array([[2.0, -3.0, 14.0, 19.0], [-3.0, 2.0, 19.0, 14.0]])
    numpy.testing.assert_array_equal(results, expected)


def test_scale_enum():
    anchor = numpy.expand_dims(numpy.array([0, 0, 16, 16]), 0)
    scales = numpy.array([0.5])
    anchor = keras.backend.variable(anchor)
    scales = keras.backend.variable(scales)
    results = keras_rcnn.backend.common._scale_enum(anchor, scales)
    results = keras.backend.eval(results)
    expected = numpy.array([[4, 4, 12, 12]])
    numpy.testing.assert_array_equal(results, expected)


def test_whctrs():
    anchor = keras.backend.cast(keras.backend.expand_dims([0, 0, 16, 16], 0), "float32")
    results0, results1, results2, results3 = keras_rcnn.backend.common._whctrs(anchor)
    results = numpy.array(
        [
            keras.backend.eval(results0),
            keras.backend.eval(results1),
            keras.backend.eval(results2),
            keras.backend.eval(results3),
        ]
    )
    expected = numpy.expand_dims([16, 16, 8, 8], 1)
    numpy.testing.assert_array_equal(results, expected)


def test_shift():
    y = keras_rcnn.backend.shift((14, 14), 16)

    assert keras.backend.eval(y).shape == (2940, 4), keras.backend.eval(y).shape

    assert y.dtype == keras.backend.floatx()


def test_bbox_transform_inv():
    anchors = 15
    features = (14, 14)
    shifted = keras_rcnn.backend.shift(features, 16)
    deltas = numpy.zeros((features[0] * features[1] * anchors, 4), numpy.float32)
    pred_boxes = keras_rcnn.backend.bbox_transform_inv(shifted, deltas)
    assert keras.backend.eval(pred_boxes).shape == (2940, 4)

    rois = numpy.array(
        [
            [55.456726, 67.949135, 86.19536, 98.13131],
            [101.945526, 0.0, 125.88675, 37.465652],
            [50.71209, 70.44628, 90.61703, 99.66291],
            [108.26308, 18.048904, 127.0, 51.715424],
            [83.864334, 0.0, 126.68436, 17.052166],
            [92.37217, 5.155298, 120.51592, 36.1372],
            [56.470436, 67.04311, 88.26943, 107.08549],
            [49.678932, 85.499756, 94.36956, 122.12128],
        ],
        numpy.float32,
    )
    deltas = numpy.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.02123459,
                0.1590581,
                0.28013495,
                -0.39532417,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.08355092,
                0.17592771,
                -0.0810278,
                -0.01217953,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.02042394,
                0.09747629,
                0.02641878,
                -0.36387,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.08266437,
                -0.2706405,
                0.15300322,
                0.09181178,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    pred_boxes = keras_rcnn.backend.bbox_transform_inv(rois, deltas)
    expected = numpy.array(
        [
            [
                55.456726,
                67.949135,
                86.19536,
                98.13131,
                51.0,
                78.0,
                93.0,
                99.0,
                55.456726,
                67.949135,
                86.19536,
                98.13131,
            ],
            [
                101.945526,
                0.0,
                125.88675,
                37.465652,
                105.0,
                7.0,
                127.0,
                44.0,
                101.945526,
                0.0,
                125.88675,
                37.465652,
            ],
            [
                50.71209,
                70.44628,
                90.61703,
                99.66291,
                51.0,
                78.0,
                93.0,
                99.0,
                50.71209,
                70.44628,
                90.61703,
                99.66291,
            ],
            [
                108.26308,
                18.048904,
                127.0,
                51.715424,
                108.26308,
                18.048904,
                127.0,
                51.715424,
                105.0,
                7.0,
                127.0,
                44.0,
            ],
            [
                83.864334,
                0.0,
                126.68436,
                17.052166,
                83.864334,
                0.0,
                126.68436,
                17.052166,
                83.864334,
                0.0,
                126.68436,
                17.052166,
            ],
            [
                92.37217,
                5.155298,
                120.51592,
                36.1372,
                92.37217,
                5.155298,
                120.51592,
                36.1372,
                92.37217,
                5.155298,
                120.51592,
                36.1372,
            ],
            [
                56.470436,
                67.04311,
                88.26943,
                107.08549,
                56.470436,
                67.04311,
                88.26943,
                107.08549,
                56.470436,
                67.04311,
                88.26943,
                107.08549,
            ],
            [
                49.678932,
                85.499756,
                94.36956,
                122.12128,
                49.678932,
                85.499756,
                94.36956,
                122.12128,
                49.678932,
                85.499756,
                94.36956,
                122.12128,
            ],
        ],
        numpy.float32,
    )
    numpy.testing.assert_array_almost_equal(
        keras.backend.eval(pred_boxes), expected, 0, verbose=True
    )


def test_smooth_l1():
    output = keras.backend.variable(
        [
            [[2.5, 0.0, 0.4, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 2.5, 0.0, 0.4]],
            [[3.5, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.9], [0.0, 0.0, 1.5, 0.0]],
        ]
    )

    target = keras.backend.zeros_like(output)

    x = keras_rcnn.backend.smooth_l1(output, target)

    numpy.testing.assert_approx_equal(keras.backend.eval(x), 8.645)

    weights = keras.backend.variable([[2, 1, 1], [0, 3, 0]])

    x = keras_rcnn.backend.smooth_l1(output, target, weights=weights)

    numpy.testing.assert_approx_equal(keras.backend.eval(x), 7.695)


def test_softmax_classification():
    output = [
        [[-100, 100, -100], [100, -100, -100], [0, 0, -100], [-100, -100, 100]],
        [[-100, 0, 0], [-100, 100, -100], [-100, 100, -100], [100, -100, -100]],
    ]

    target = [
        [[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
    ]

    weights = [[1.0, 1.0, 0.5, 1.0], [1.0, 1.0, 1.0, 0.0]]

    x = keras_rcnn.backend.softmax_classification(
        keras.backend.variable(target),
        keras.backend.variable(output),
        weights=keras.backend.variable(weights),
    )

    output_y = numpy.reshape(output, [-1, 3])
    target_y = numpy.reshape(target, [-1, 3])
    output_y = output_y / numpy.sum(output_y, axis=1, keepdims=True)

    # epsilon = keras.backend.epsilon()
    # stop python for complaining weakref
    epsilon = 1e-07

    output_y = numpy.clip(output_y, epsilon, 1.0 - epsilon)
    _y = -numpy.sum(target_y * numpy.log(output_y), axis=1)
    y = _y * numpy.reshape(weights, -1)

    numpy.testing.assert_array_almost_equal(keras.backend.eval(x), y)

    x = keras_rcnn.backend.softmax_classification(
        keras.backend.variable(target),
        keras.backend.variable(output),
        anchored=True,
        weights=keras.backend.variable(weights),
    )

    y = weights * numpy.reshape(_y, numpy.asarray(weights).shape)
    numpy.testing.assert_array_almost_equal(keras.backend.eval(x), y)
