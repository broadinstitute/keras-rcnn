import keras.backend
import keras.layers
import keras.models
import numpy
import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers
import keras_rcnn.layers.object_detection._anchor_target as anchor_target


def test_label():
    stride = 16
    feat_h, feat_w = (14, 14)
    img_info = keras.backend.variable([[224, 224, 3]])

    gt_boxes = keras.backend.variable(100 * numpy.random.random((91, 4)))
    gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)

    all_bbox = keras_rcnn.backend.shift((feat_h, feat_w), stride)

    inds_inside, all_inside_bbox = anchor_target.inside_image(
        all_bbox, img_info[0])

    argmax_overlaps_inds, bbox_labels = anchor_target.label(
        gt_boxes, all_inside_bbox, inds_inside)

    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (376,)

    assert result2.shape == (376,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1

    argmax_overlaps_inds, bbox_labels = anchor_target.label(
        gt_boxes, all_inside_bbox, inds_inside, clobber_positives=False)

    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (376,)

    assert result2.shape == (376,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1

    gt_boxes = keras.backend.variable(224 * numpy.random.random((55, 4)))
    gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)
    argmax_overlaps_inds, bbox_labels = anchor_target.label(
        gt_boxes, all_inside_bbox, inds_inside, clobber_positives=False)
    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (376,)

    assert result2.shape == (376,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1


def test_subsample_positive_labels():
    x = keras.backend.ones((10,))

    y = anchor_target.subsample_positive_labels(
        x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((1000,))

    y = anchor_target.subsample_positive_labels(
        x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_subsample_negative_labels():
    x = keras.backend.zeros((10,))

    y = anchor_target.subsample_negative_labels(
        x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.zeros((1000,))

    y = anchor_target.subsample_negative_labels(
        x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_balance():
    x = keras.backend.zeros((91,))

    y = anchor_target.balance(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((91,))

    y = anchor_target.balance(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((1000,))

    y = anchor_target.balance(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_overlapping():
    stride = 16
    features = (14, 14)
    img_info = keras.backend.variable([[224, 224, 3]])
    gt_boxes = numpy.zeros((91, 4))
    gt_boxes = keras.backend.variable(gt_boxes)
    img_info = img_info[0]

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = anchor_target.inside_image(
        all_anchors, img_info)

    a, max_overlaps, gt_argmax_overlaps_inds = anchor_target.overlapping(
        all_inside_anchors, gt_boxes, inds_inside)

    a = keras.backend.eval(a)
    max_overlaps = keras.backend.eval(max_overlaps)
    gt_argmax_overlaps_inds = keras.backend.eval(gt_argmax_overlaps_inds)

    assert a.shape == (376,)

    assert max_overlaps.shape == (376,)

    assert gt_argmax_overlaps_inds.shape == (91,)


def test_unmap():
    stride = 16
    features = (14, 14)
    anchors = 9
    total_anchors = features[0] * features[1] * anchors
    img_info = keras.backend.variable([[224, 224, 3]])
    gt_boxes = numpy.zeros((91, 4))
    gt_boxes = keras.backend.variable(gt_boxes)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = anchor_target.inside_image(
        all_anchors, img_info[0])

    argmax_overlaps_indices, labels = anchor_target.label(
        gt_boxes, all_inside_anchors, inds_inside)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(
        all_inside_anchors,
        keras.backend.gather(gt_boxes, argmax_overlaps_indices)
    )

    labels = anchor_target.unmap(labels,
                                 total_anchors,
                                 inds_inside,
                                 fill=-1)
    bbox_reg_targets = anchor_target.unmap(
        bbox_reg_targets, total_anchors, inds_inside, fill=0)

    assert keras.backend.eval(labels).shape == (total_anchors,)
    assert keras.backend.eval(bbox_reg_targets).shape == (total_anchors, 4)


def test_inside_image():
    stride = 16
    features = (14, 14)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    img_info = (224, 224, 1)

    inds_inside, all_inside_anchors = anchor_target.inside_image(
        all_anchors, img_info)

    inds_inside = keras.backend.eval(inds_inside)

    assert inds_inside.shape == (376,)

    all_inside_anchors = keras.backend.eval(all_inside_anchors)

    assert all_inside_anchors.shape == (376, 4)


def test_inside_and_outside_weights_1():
    anchors = numpy.array(
        [[30, 20, 50, 30],
         [10, 15, 20, 25],
         [ 5, 15, 20, 22]]
    )

    anchors = keras.backend.constant(anchors)

    subsample = keras.backend.constant([1, 0, 1])

    positive_weight = -1.0

    proposed_inside_weights = [1.0, 0.5, 0.7, 1.0]

    target_inside_weights = numpy.array(
        [[1.0, 0.5, 0.7, 1.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 0.5, 0.7, 1.0]]
    )

    target_inside_weights = keras.backend.constant(target_inside_weights)

    target_outside_weights = keras.backend.ones_like(anchors, dtype=keras.backend.floatx())
    target_outside_weights /= target_outside_weights

    output_inside_weights, output_outside_weights = anchor_target.inside_and_outside_weights(anchors, subsample, positive_weight, proposed_inside_weights)

    target_inside_weights = keras.backend.eval(target_inside_weights)
    output_inside_weights = keras.backend.eval(output_inside_weights)

    numpy.testing.assert_array_equal(target_inside_weights, output_inside_weights)

    target_outside_weights = keras.backend.eval(target_outside_weights)
    output_outside_weights = keras.backend.eval(output_outside_weights)

    numpy.testing.assert_array_equal(target_outside_weights, output_outside_weights)


def test_inside_and_outside_weights_2():
    anchors = numpy.array(
        [[30, 20, 50, 30],
         [10, 15, 20, 25],
         [ 5, 15, 20, 22]]
    )

    anchors = keras.backend.constant(anchors)

    subsample = keras.backend.constant([1, 0, 1])

    positive_weight = 0.6

    proposed_inside_weights = [1.0, 0.5, 0.7, 1.0]

    target_inside_weights = numpy.array(
        [[1.0, 0.5, 0.7, 1.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 0.5, 0.7, 1.0]]
    )

    target_inside_weights = keras.backend.constant(target_inside_weights)

    target_outside_weights = numpy.array(
        [[0.3, 0.3, 0.3, 0.3],
         [0.4, 0.4, 0.4, 0.4],
         [0.3, 0.3, 0.3, 0.3]]
    )

    target_outside_weights = keras.backend.constant(target_outside_weights)

    output_inside_weights, output_outside_weights = anchor_target.inside_and_outside_weights(anchors, subsample, positive_weight, proposed_inside_weights)

    target_inside_weights = keras.backend.eval(target_inside_weights)
    output_inside_weights = keras.backend.eval(output_inside_weights)

    numpy.testing.assert_array_equal(target_inside_weights, output_inside_weights)

    target_outside_weights = keras.backend.eval(target_outside_weights)
    output_outside_weights = keras.backend.eval(output_outside_weights)

    numpy.testing.assert_array_equal(target_outside_weights, output_outside_weights)
