import keras.backend
import keras_rcnn.backend.common
import numpy
import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers.object_detection._object_proposal


def test_shuffle():
    x = keras.backend.variable(numpy.random.random((10,)))

    keras_rcnn.backend.shuffle(x)


def test_label():
    stride = 16
    feat_h, feat_w = (14, 14)
    img_info = keras.backend.variable([[224, 224, 3]])

    gt_boxes = keras.backend.variable(100 * numpy.random.random((91, 4)))
    gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)

    all_bbox = keras_rcnn.backend.shift((feat_h, feat_w), stride)

    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(all_bbox,
                                                                   img_info[0])

    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(gt_boxes,
                                                                 all_inside_bbox,
                                                                 inds_inside)

    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (84,)

    assert result2.shape == (84,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1

    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(gt_boxes, all_inside_bbox, inds_inside, clobber_positives=False)

    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (84,)

    assert result2.shape == (84,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1

    gt_boxes = keras.backend.variable(224 * numpy.random.random((55, 4)))
    gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)
    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(gt_boxes, all_inside_bbox, inds_inside, clobber_positives=False)
    result1 = keras.backend.eval(argmax_overlaps_inds)
    result2 = keras.backend.eval(bbox_labels)

    assert result1.shape == (84,)

    assert result2.shape == (84,)

    assert numpy.max(result2) <= 1

    assert numpy.min(result2) >= -1



def test_non_max_suppression():
    boxes = numpy.zeros((1764, 4))
    scores = numpy.random.rand(14 * 14, 9).flatten()
    threshold = 0.5
    maximum = 100
    nms = tensorflow.image.non_max_suppression(boxes=boxes,
                                               iou_threshold=threshold,
                                               max_output_size=maximum,
                                               scores=scores)
    assert keras.backend.eval(nms).shape == (maximum,)


def test_bbox_transform_inv():
    anchors = 9
    features = (14, 14)
    shifted = keras_rcnn.backend.shift(features, 16)
    boxes = numpy.zeros((features[0] * features[1] * anchors, 4))
    boxes = keras.backend.variable(boxes)
    pred_boxes = keras_rcnn.backend.bbox_transform_inv(shifted, boxes)
    assert keras.backend.eval(pred_boxes).shape == (1764, 4)


def test_resize_images():
    im = numpy.zeros((1200, 1600, 3))
    shape = (256, 256)
    im = keras.backend.variable(im)
    resize = keras_rcnn.backend.resize_images(im, shape)
    assert keras.backend.eval(resize).shape == (256, 256, 3)


def test_subsample_positive_labels():
    x = keras.backend.ones((10,))

    y = keras_rcnn.backend.subsample_positive_labels(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((1000,))

    y = keras_rcnn.backend.subsample_positive_labels(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_subsample_negative_labels():
    x = keras.backend.zeros((10,))

    y = keras_rcnn.backend.subsample_negative_labels(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.zeros((1000,))

    y = keras_rcnn.backend.subsample_negative_labels(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_balance():
    x = keras.backend.zeros((91,))

    y = keras_rcnn.backend.balance(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((91,))

    y = keras_rcnn.backend.balance(x)

    numpy.testing.assert_array_equal(
        keras.backend.eval(x),
        keras.backend.eval(y)
    )

    x = keras.backend.ones((1000,))

    y = keras_rcnn.backend.balance(x)

    assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))


def test_crop_and_resize():
    image = keras.backend.variable(numpy.ones((1, 28, 28, 3)))

    boxes = keras.backend.variable(
        numpy.array([[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.8, 0.8]]]))

    size = [7, 7]

    slices = keras_rcnn.backend.crop_and_resize(image, boxes, size)

    assert keras.backend.eval(slices).shape == (2, 7, 7, 3)


def test_overlapping():
    stride = 16
    features = (14, 14)
    img_info = keras.backend.variable([[224, 224, 3]])
    gt_boxes = numpy.zeros((91, 4))
    gt_boxes = keras.backend.variable(gt_boxes)
    img_info = img_info[0]

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(
        all_anchors, img_info)

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = keras_rcnn.backend.overlapping(
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

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(all_anchors, img_info[0])

    argmax_overlaps_indices, labels = keras_rcnn.backend.label(gt_boxes, all_inside_anchors, inds_inside)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_anchors, keras.backend.gather(gt_boxes, argmax_overlaps_indices))

    labels = keras_rcnn.backend.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_reg_targets = keras_rcnn.backend.unmap(bbox_reg_targets, total_anchors, inds_inside, fill=0)

    assert keras.backend.eval(labels).shape == (total_anchors, )
    assert keras.backend.eval(bbox_reg_targets).shape == (total_anchors, 4)


def test_get_bbox_regression_labels():
    N = 10
    bbox_target_data = keras.backend.zeros((N, 4))
    num_classes = 3
    labels = numpy.reshape([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], (1, -1, 3))
    labels = keras.backend.variable(labels)
    bbox_targets = keras_rcnn.backend.tensorflow_backend.get_bbox_regression_labels(labels, bbox_target_data)
    bbox_targets = keras.backend.eval(bbox_targets)

    assert bbox_targets.shape == (N, 4 * num_classes)


def test_sample_rois():
    N = 5
    gt_boxes = numpy.zeros((N, 4))
    gt_boxes = keras.backend.variable(gt_boxes)
    num_classes = 3
    gt_labels = numpy.reshape([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], (-1, 3))
    gt_labels = keras.backend.variable(gt_labels)

    fg_thresh = 0.7

    fg_fraction = 0.5
    batchsize = 256
    num_images = 1
    bg_thresh_lo = 0.1
    bg_thresh_hi = 0.5
    N_proposals = 200
    all_rois = keras.backend.zeros((N_proposals, 4))

    rois_per_image = batchsize // num_images
    fg_rois_per_image = int(fg_fraction * rois_per_image)
    rois, labels, bbox_targets = keras_rcnn.backend.sample_rois(all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image, fg_thresh, bg_thresh_hi, bg_thresh_lo)
    assert keras.backend.eval(labels).shape == (N_proposals, num_classes)
    assert keras.backend.eval(rois).shape == (N_proposals, 4)
    assert keras.backend.eval(bbox_targets).shape == (N_proposals, 4*num_classes)
