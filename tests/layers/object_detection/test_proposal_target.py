import keras.backend
import keras.utils.test_utils
import numpy

import keras_rcnn.backend
import keras_rcnn.layers


def test_get_config():
    layer = keras_rcnn.layers.ProposalTarget()

    expected = {
        "background_threshold": (0.1, 0.5),
        "foreground": 0.5,
        "foreground_threshold": (0.5, 1.0),
        "maximum_proposals": 32,
        "name": 'proposal_target_1',
        "trainable": True
    }

    assert layer.get_config() == expected


def test_proposal_target():
    keras.backend.set_learning_phase(True)

    a = keras.layers.Input((None, 4))
    b = keras.layers.Input((None, 2))
    c = keras.layers.Input((None, 4))

    layer = keras_rcnn.layers.ProposalTarget()

    x, y, z = layer([a, b, c])

    model = keras.models.Model([a, b, c], [x, y, z])

    assert model.output_shape == [(1, None, 4), (1, None, 2), (1, None, 8)]

    a = numpy.random.random((1, 300, 4))
    b = numpy.random.random((1, 100, 2))
    c = numpy.random.random((1, 100, 4))

    x, y, z = model.predict([a, b, c])

    assert x.shape == (1, 32, 4)
    assert y.shape == (1, 32, 2)
    assert z.shape == (1, 32, 8)


def test_get_bbox_regression_labels():
    p = keras_rcnn.layers.ProposalTarget()

    bbox_target_data = numpy.array([
        [0, 0, 0, 0],
        [.1, .2, -.1, -.3],
        [1, 2, 3, 4],
        [.1, .2, -.1, -.3],
        [1, 2, 3, 4]
    ])
    bbox_target_data = keras.backend.cast(bbox_target_data, dtype=keras.backend.floatx())
    num_classes = 3
    labels = numpy.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    labels = numpy.argmax(labels, axis=1)
    bbox_targets = p.get_bbox_regression_labels(bbox_target_data, labels, num_classes)
    bbox_targets = keras.backend.eval(bbox_targets)

    assert bbox_targets.shape == (5, 4 * num_classes)
    expected = numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, .1, .2, -.1, -.3],
        [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0]
    ])
    numpy.testing.assert_array_almost_equal(bbox_targets, expected)


def test_get_bbox_targets():
    p = keras_rcnn.layers.ProposalTarget()

    rois = numpy.array([[0, 14.4, 30.1, 99.], [40., 3.2, 55.5, 33.7], [30.7, 1.2, 66.5, 23.8], [50.7, 50.2, 86.5, 63.8]])
    gt_boxes = numpy.array([[3., 20., 22., 100.], [40., 3., 55., 33.], [30., 1., 66., 24.], [15., 18., 59., 59.]])
    gt_boxes = keras.backend.cast(gt_boxes, dtype=keras.backend.floatx())
    num_classes = 4
    labels = numpy.array([1, 1, 2, 1])

    bbox_targets = p.get_bbox_targets(rois, gt_boxes, labels, num_classes)
    bbox_targets = keras.backend.eval(bbox_targets)
    expected = numpy.array([
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          -0.08471760797342194, 0.039007092198581526, -0.46008619258838973, -0.05590763193829595,
          0.        ,  0.        , 0.        ,  0.        ,
          0.        ,  0.        ,  0.        , 0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          -0.016129032258064516, -0.014754098360655828, -0.03278982282299084, -0.016529301951210697,
          0.        ,  0.        , 0.        ,  0.        ,
          0.        ,  0.        ,  0.        , 0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          0.        , 0.        ,  0.        ,  0.        ,
          -0.01675977653631289, 0.0, 0.0055710450494554295, 0.017544309650909525,
          0.        ,  0.        ,  0.        , 0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          -0.88268156424581, -1.3602941176470593, 0.20624174051160657, 1.103502273962302,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ]])
    numpy.testing.assert_array_almost_equal(bbox_targets, expected)


def test_sample_indices():
    p = keras_rcnn.layers.ProposalTarget()
    indices = numpy.array([0, 3, 4, 10, 22])
    sampled = p.sample_indices(indices, 3)
    sampled = keras.backend.eval(sampled)
    assert sampled.shape == (3,)
    for i in sampled:
        assert i in indices

    sampled = p.sample_indices(indices, 4)
    sampled = keras.backend.eval(sampled)
    assert sampled.shape == (4,)
    for i in sampled:
        assert i in indices
