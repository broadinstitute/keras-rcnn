import keras.backend
import keras_rcnn.layers
import numpy
import keras_rcnn.layers.object_detection

def test_call_classification():
    anchors = 9
    layer = keras_rcnn.layers.RPNClassificationLoss(anchors=anchors)
    y_true = keras.backend.variable(100 * numpy.random.random((1, 91, 4)))
    scores = keras.backend.variable(0.5 * numpy.ones((1, 14, 14, anchors * 2)))
    metadata = keras.backend.variable(numpy.array([[224, 224, 1]]))

    labels, bbox_reg_targets = keras_rcnn.layers.AnchorTarget()([scores, y_true, metadata])
    numpy.testing.assert_array_equal(layer.call([scores, labels]), scores)

    assert len(layer.losses) == 1

    expected_loss = -numpy.log(0.5)
    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)


def test_call_regression():
    anchors = 9
    metadata = keras.backend.variable(numpy.array([[224, 224, 1]]))
    layer = keras_rcnn.layers.RPNRegressionLoss(anchors=anchors)
    rr, cc = 14, 14
    stride = 16
    all_anchors = keras_rcnn.backend.shift((rr, cc), stride)
    # only keep anchors inside the image
    inds_inside, y_true = keras_rcnn.layers.object_detection._anchor_target.inside_image(all_anchors, metadata[0])

    scores = keras.backend.variable(numpy.zeros((1, 14, 14, anchors * 2)))
    deltas = keras.backend.variable(numpy.zeros((1, 14, 14, anchors * 4)))

    expected_loss = 0

    labels, bbox_reg_targets = keras_rcnn.layers.AnchorTarget()([scores, keras.backend.expand_dims(y_true, 0), metadata])
    numpy.testing.assert_array_equal(
        layer.call([deltas, bbox_reg_targets, labels]), deltas)

    assert len(layer.losses) == 1

    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)
