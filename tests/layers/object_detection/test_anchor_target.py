import keras.backend
import keras.layers
import keras.models
import numpy
import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers
import keras_rcnn.layers.object_detection._anchor_target


# class TestAnchorTarget:
#     def test_call(self):
#         gt_boxes = keras.backend.variable(numpy.random.random((1, 10000, 4)))
#         image = keras.layers.Input((224, 224, 3))
#         scores = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 2)))
#
#         proposal_target = keras_rcnn.layers.AnchorTarget()
#
#         proposal_target.call([scores, gt_boxes, image])

def test_label():
    stride = 16
    feat_h, feat_w = (14, 14)
    img_info = keras.backend.variable([[224, 224, 3]])

    gt_boxes = keras.backend.variable(100 * numpy.random.random((91, 4)))
    gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)

    all_bbox = keras_rcnn.backend.shift((feat_h, feat_w), stride)

    inds_inside, all_inside_bbox = keras_rcnn.layers.object_detection._anchor_target.inside_image(all_bbox, img_info[0])

    argmax_overlaps_inds, bbox_labels = keras_rcnn.layers.object_detection._anchor_target.label(gt_boxes, all_inside_bbox, inds_inside)

    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (84,)

    assert result2.shape == (84,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1

    argmax_overlaps_inds, bbox_labels = keras_rcnn.layers.object_detection._anchor_target.label(gt_boxes, all_inside_bbox, inds_inside, clobber_positives=False)

    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (84,)

    assert result2.shape == (84,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1

    gt_boxes = keras.backend.variable(224 * numpy.random.random((55, 4)))
    gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)
    argmax_overlaps_inds, bbox_labels = keras_rcnn.layers.object_detection._anchor_target.label(gt_boxes, all_inside_bbox, inds_inside, clobber_positives=False)
    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (84,)

    assert result2.shape == (84,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1


def test_subsample_positive_labels():
    x = keras.backend.ones((10,))

    y = keras_rcnn.layers.object_detection._anchor_target.subsample_positive_labels(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((1000,))

    y = keras_rcnn.layers.object_detection._anchor_target.subsample_positive_labels(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_subsample_negative_labels():
    x = keras.backend.zeros((10,))

    y = keras_rcnn.layers.object_detection._anchor_target.subsample_negative_labels(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.zeros((1000,))

    y = keras_rcnn.layers.object_detection._anchor_target.subsample_negative_labels(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_balance():
    x = keras.backend.zeros((91,))

    y = keras_rcnn.layers.object_detection._anchor_target.balance(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((91,))

    y = keras_rcnn.layers.object_detection._anchor_target.balance(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((1000,))

    y = keras_rcnn.layers.object_detection._anchor_target.balance(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_overlapping():
    stride = 16
    features = (14, 14)
    img_info = keras.backend.variable([[224, 224, 3]])
    gt_boxes = numpy.zeros((91, 4))
    gt_boxes = keras.backend.variable(gt_boxes)
    img_info = img_info[0]

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = keras_rcnn.layers.object_detection._anchor_target.inside_image(
        all_anchors, img_info)

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = keras_rcnn.layers.object_detection._anchor_target.overlapping(
        all_inside_anchors, gt_boxes, inds_inside)

    argmax_overlaps_inds = keras.backend.eval(argmax_overlaps_inds)
    max_overlaps = keras.backend.eval(max_overlaps)
    gt_argmax_overlaps_inds = keras.backend.eval(gt_argmax_overlaps_inds)

    assert argmax_overlaps_inds.shape == (84,)

    assert max_overlaps.shape == (84,)

    assert gt_argmax_overlaps_inds.shape == (91,)


def test_unmap():
    stride = 16
    features = (14, 14)
    anchors = 9
    total_anchors = features[0]*features[1]*anchors
    img_info = keras.backend.variable([[224, 224, 3]])
    gt_boxes = numpy.zeros((91, 4))
    gt_boxes = keras.backend.variable(gt_boxes)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = keras_rcnn.layers.object_detection._anchor_target.inside_image(all_anchors, img_info[0])

    argmax_overlaps_indices, labels = keras_rcnn.layers.object_detection._anchor_target.label(gt_boxes, all_inside_anchors, inds_inside)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_anchors, keras.backend.gather(gt_boxes, argmax_overlaps_indices))

    labels = keras_rcnn.layers.object_detection._anchor_target.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_reg_targets = keras_rcnn.layers.object_detection._anchor_target.unmap(bbox_reg_targets, total_anchors, inds_inside, fill=0)

    assert keras.backend.eval(labels).shape == (total_anchors, )
    assert keras.backend.eval(bbox_reg_targets).shape == (total_anchors, 4)


def test_inside_image():
    stride = 16
    features = (14, 14)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    img_info = (224, 224, 1)

    inds_inside, all_inside_anchors = keras_rcnn.layers.object_detection._anchor_target.inside_image(all_anchors, img_info)

    inds_inside = keras.backend.eval(inds_inside)

    assert inds_inside.shape == (84,)

    all_inside_anchors = keras.backend.eval(all_inside_anchors)

    assert all_inside_anchors.shape == (84, 4)
