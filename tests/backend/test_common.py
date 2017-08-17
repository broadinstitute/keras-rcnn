import keras.backend
import numpy
import numpy.testing

import keras_rcnn.backend
import keras_rcnn.backend.common


def test_anchor():
    x = numpy.array(
      [[ -84.,  -40.,  99.,  55.],
       [-176.,  -88., 191., 103.],
       [-360., -184., 375., 199.],
       [ -56.,  -56.,  71.,  71.],
       [-120., -120., 135., 135.],
       [-248., -248., 263., 263.],
       [ -36.,  -80.,  51.,  95.],
       [ -80., -168.,  95., 183.],
       [-168., -344., 183., 359.]]
    )
    y = keras_rcnn.backend.anchor()
    y = keras.backend.eval(y)
    numpy.testing.assert_array_almost_equal(x, y)


def test_clip():
    boxes = numpy.array([[0,0,0,0], [1, 2, 3, 4], [-4, 2, 1000, 6000], [3, -10, 223, 224]])
    shape = [224,224]
    boxes = keras.backend.variable(boxes)
    results = keras_rcnn.backend.clip(boxes, shape)
    results = keras.backend.eval(results)
    expected = numpy.array([[0,0,0,0], [1,2,3,4], [0,2,223,223], [3, 0, 223, 223]])
    numpy.testing.assert_array_almost_equal(results, expected)


def test_bbox_transform():
    gt_rois = numpy.array([[ -84., -40., 99., 55.], [-176., -88., 191., 103.], [-360., -184., 375., 199.], [ -56., -56., 71., 71.], [-120., -120., 135., 135.], [-248., -248., 263., 263.], [ -36., -80., 51., 95.], [ -80., -168., 95., 183.], [-168., -344., 183., 359.]])
    ex_rois = 2 * gt_rois
    gt_rois = keras.backend.variable(gt_rois)
    ex_rois = keras.backend.variable(ex_rois)
    results = keras_rcnn.backend.bbox_transform(ex_rois, gt_rois)
    results = keras.backend.eval(results)
    expected = numpy.array([[-0.02043597, -0.03926702, -0.69042609, -0.68792524], [-0.01020408, -0.01958225, -0.69178756, -0.69053962], [-0.00509857, -0.00977836, -0.6924676 , -0.69184425], [-0.02941176, -0.02941176, -0.68923328, -0.68923328], [-0.0146771 , -0.0146771 , -0.69119215, -0.69119215], [-0.00733138, -0.00733138, -0.69217014, -0.69217014], [-0.04285714, -0.02136752, -0.68744916, -0.69030223], [-0.02136752, -0.01066856, -0.69030223, -0.69172572], [-0.01066856, -0.00533049, -0.69172572, -0.6924367 ]])
    numpy.testing.assert_array_almost_equal(results, expected)


def test_mkanchors():
    ws = numpy.array([1,2,3])
    hs = numpy.array([4,5,6])
    x_ctr = keras.backend.variable([1], 'float32')
    y_ctr = keras.backend.variable([2], 'float32')
    ws = keras.backend.variable(ws, 'float32')
    hs = keras.backend.variable(hs, 'float32')
    results = keras_rcnn.backend.common._mkanchors(ws, hs, x_ctr, y_ctr)
    results = keras.backend.eval(results)
    expected = numpy.array([[1, 0.5, 1, 3.5], [0.5, 0, 1.5, 4 ] , [0, -0.5, 2, 4.5] ])
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
    expected = numpy.array([[ 0. , 0. , 0. , 0. ], [ 0. , -0.5, 0. , 0.5], [ 0. , -1. , 0. , 1. ]])
    numpy.testing.assert_array_equal(results, expected)
    anchor = numpy.expand_dims(numpy.array([2, 3, 100, 100]), 0)
    anchor = keras.backend.variable(anchor)
    results = keras_rcnn.backend.common._ratio_enum(anchor, ratios)
    results = keras.backend.eval(results)
    expected = numpy.array([[ 2.5, 3. , 99.5, 100. ], [ 16.5, -18. , 85.5, 121. ], [ 23. , -33.5, 79. , 136.5]])
    numpy.testing.assert_array_equal(results, expected)


def test_scale_enum():
    anchor = numpy.expand_dims(numpy.array([0, 0, 0, 0]), 0)
    scales = numpy.array([1, 2, 3])
    anchor = keras.backend.variable(anchor)
    scales = keras.backend.variable(scales)
    results = keras_rcnn.backend.common._scale_enum(anchor, scales)
    results = keras.backend.eval(results)
    expected = numpy.array([[0, 0, 0, 0], [-0.5, -0.5, 0.5, 0.5], [-1. , -1. , 1. , 1. ]])
    numpy.testing.assert_array_equal(results, expected)
    anchor = keras.backend.cast(numpy.expand_dims(numpy.array([2, 3, 100, 100]), 0), 'float32')
    anchor = keras.backend.variable(anchor)
    results = keras_rcnn.backend.common._scale_enum(anchor, scales)
    results = keras.backend.eval(results)
    expected = numpy.array([[ 2. , 3. , 100. , 100. ], [ -47.5, -46. , 149.5, 149. ], [ -97. , -95. , 199. , 198. ]])
    numpy.testing.assert_array_equal(results, expected)


def test_whctrs():
    anchor = keras.backend.cast(keras.backend.expand_dims([0,0,0,0], 0), 'float32')
    results0, results1, results2, results3 = keras_rcnn.backend.common._whctrs(anchor)
    results = numpy.array([keras.backend.eval(results0), keras.backend.eval(results1), keras.backend.eval(results2), keras.backend.eval(results3)])
    expected = numpy.expand_dims([1,1,0,0], 1)
    numpy.testing.assert_array_equal(results, expected)
    anchor = keras.backend.cast(keras.backend.expand_dims([2, 3, 100, 100], 0), 'float32')
    results0, results1, results2, results3 = keras_rcnn.backend.common._whctrs(anchor)
    results = numpy.array([keras.backend.eval(results0), keras.backend.eval(results1), keras.backend.eval(results2), keras.backend.eval(results3)])
    expected = numpy.expand_dims([99, 98, 51, 51.5], 1)
    numpy.testing.assert_array_equal(results, expected)


def test_shift():
    y = keras_rcnn.backend.shift((14, 14), 16)

    assert keras.backend.int_shape(y) == (1764, 4)

    assert y.dtype == keras.backend.floatx()


def test_inside_image():
    stride = 16
    features = (14, 14)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    img_info = (224, 224, 1)

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(all_anchors, img_info)

    inds_inside = keras.backend.eval(inds_inside)

    assert inds_inside.shape == (84,)

    all_inside_anchors = keras.backend.eval(all_inside_anchors)

    assert all_inside_anchors.shape == (84, 4)


def test_filter_boxes():
    proposals = numpy.array(
        [[0, 2, 3, 10],
         [-1, -5, 4, 8],
         [0, 0, 1, 1]]
    )

    minimum = 3

    results = keras_rcnn.backend.filter_boxes(proposals, minimum)

    numpy.testing.assert_array_equal(keras.backend.eval(results), numpy.array([0, 1]))
