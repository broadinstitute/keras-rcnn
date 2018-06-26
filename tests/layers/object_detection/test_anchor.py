import keras.backend
import keras.layers
import keras.models
import numpy
import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers
import keras_rcnn.layers.object_detection._anchor as anchor_target


class TestAnchor:
    def test_call(self):
        target_bounding_boxes = numpy.random.random((1, 10, 4))
        target_bounding_boxes = keras.backend.variable(target_bounding_boxes)

        target_metadata = keras.backend.variable([[224, 224, 1]])

        output_scores = numpy.random.random((1, 14, 14, 9 * 2))
        output_scores = keras.backend.variable(output_scores)

        _, _, _ = keras_rcnn.layers.Anchor(
            padding=1
        )([
            target_bounding_boxes,
            target_metadata,
            output_scores
        ])

    def test_label(self, anchor_layer):
        stride = 16
        feat_h, feat_w = (7, 7)
        img_info = keras.backend.variable([[112, 112, 3]])

        gt_boxes = keras.backend.variable(100 * numpy.random.random((91, 4)))
        gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)

        all_anchors = keras_rcnn.backend.shift((feat_h, feat_w), stride)

        anchor_layer.padding = 1
        anchor_layer.metadata = img_info[0]

        inds_inside, all_inside_anchors = anchor_layer._inside_image(all_anchors)

        all_inside_anchors = keras_rcnn.backend.clip(all_inside_anchors, img_info[0][:2])

        argmax_overlaps_inds, anchor_labels = anchor_layer._label(gt_boxes, all_inside_anchors, inds_inside)

        result1 = keras.backend.eval(argmax_overlaps_inds)
        result2 = keras.backend.eval(anchor_labels)

        assert result1.shape == (224,), keras.backend.eval(inds_inside).shape

        assert result2.shape == (224,)

        assert numpy.max(result2) <= 1

        assert numpy.min(result2) >= -1

        argmax_overlaps_inds, anchor_labels = anchor_layer._label(gt_boxes, all_inside_anchors, inds_inside)

        result1 = keras.backend.eval(argmax_overlaps_inds)
        result2 = keras.backend.eval(anchor_labels)

        assert result1.shape == (224,)

        assert result2.shape == (224,)

        assert numpy.max(result2) <= 1

        assert numpy.min(result2) >= -1

        gt_boxes = keras.backend.variable(224 * numpy.random.random((55, 4)))
        gt_boxes = tensorflow.convert_to_tensor(gt_boxes, dtype=tensorflow.float32)

        argmax_overlaps_inds, anchor_labels = anchor_layer._label(gt_boxes, all_inside_anchors, inds_inside)

        result1 = keras.backend.eval(argmax_overlaps_inds)
        result2 = keras.backend.eval(anchor_labels)

        assert result1.shape == (224,)

        assert result2.shape == (224,)

        assert numpy.max(result2) <= 1

        assert numpy.min(result2) >= -1

    def test_subsample_positive_labels(self, anchor_layer):
        x = keras.backend.ones((10,))

        y = anchor_layer._subsample_positive_labels(
            x)

        numpy.testing.assert_array_equal(
            keras.backend.eval(x),
            keras.backend.eval(y)
        )

        x = keras.backend.ones((1000,))

        y = anchor_layer._subsample_positive_labels(
            x)

        assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))

    def test_subsample_negative_labels(self, anchor_layer):
        x = keras.backend.zeros((10,))

        y = anchor_layer._subsample_negative_labels(x)

        numpy.testing.assert_array_equal(
            keras.backend.eval(x),
            keras.backend.eval(y)
        )

        x = keras.backend.zeros((1000,))

        y = anchor_layer._subsample_negative_labels(x)

        assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))

    def test_balance(self, anchor_layer):
        x = keras.backend.zeros((91,))

        y = anchor_layer._balance(x)

        numpy.testing.assert_array_equal(
            keras.backend.eval(x),
            keras.backend.eval(y)
        )

        x = keras.backend.ones((91,))

        y = anchor_layer._balance(x)

        numpy.testing.assert_array_equal(
            keras.backend.eval(x),
            keras.backend.eval(y)
        )

        x = keras.backend.ones((1000,))

        y = anchor_layer._balance(x)

        assert keras.backend.eval(keras.backend.sum(y) < keras.backend.sum(x))

    def test_overlapping(self, anchor_layer):
        stride = 16
        features = (7, 7)
        img_info = keras.backend.variable([[112, 112, 3]])
        gt_boxes = numpy.zeros((91, 4))
        gt_boxes = keras.backend.variable(gt_boxes)

        all_anchors = keras_rcnn.backend.shift(features, stride)

        anchor_layer.padding = 1
        anchor_layer.metadata = img_info[0]

        inds_inside, all_inside_anchors = anchor_layer._inside_image(all_anchors)

        all_inside_anchors = keras_rcnn.backend.clip(all_inside_anchors, img_info[0][:2])

        a, max_overlaps, gt_argmax_overlaps_inds = anchor_layer._overlapping(all_inside_anchors, gt_boxes, inds_inside)

        a = keras.backend.eval(a)
        max_overlaps = keras.backend.eval(max_overlaps)
        gt_argmax_overlaps_inds = keras.backend.eval(gt_argmax_overlaps_inds)

        assert a.shape == (224,)

        assert max_overlaps.shape == (224,)

        assert gt_argmax_overlaps_inds.shape == (91,)

    def test_unmap(self, anchor_layer):
        stride = 16
        features = (14, 14)
        anchors = 15
        total_anchors = features[0] * features[1] * anchors
        img_info = keras.backend.variable([[224, 224, 3]])
        gt_boxes = numpy.zeros((91, 4))
        gt_boxes = keras.backend.variable(gt_boxes)

        all_anchors = keras_rcnn.backend.shift(features, stride)

        anchor_layer.padding = 1
        anchor_layer.metadata = img_info[0]
        anchor_layer.k = total_anchors

        inds_inside, all_inside_anchors = anchor_layer._inside_image(all_anchors)

        all_inside_anchors = keras_rcnn.backend.clip(all_inside_anchors, img_info[0][:2])

        argmax_overlaps_indices, labels = anchor_layer._label(gt_boxes, all_inside_anchors, inds_inside)

        bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_anchors, keras.backend.gather(gt_boxes, argmax_overlaps_indices))

        labels = anchor_layer._unmap(labels, inds_inside, fill=-1)

        bbox_reg_targets = anchor_layer._unmap(bbox_reg_targets, inds_inside, fill=0)

        assert keras.backend.eval(labels).shape == (total_anchors,)
        assert keras.backend.eval(bbox_reg_targets).shape == (total_anchors, 4)

    def test_inside_image(self, anchor_layer):
        stride = 16
        features = (7, 7)

        all_anchors = keras_rcnn.backend.shift(features, stride)

        img_info = numpy.array([[112, 112, 1]])

        anchor_layer.metadata = img_info[0]
        anchor_layer.padding = 1

        inds_inside, all_inside_anchors = anchor_layer._inside_image(all_anchors)

        all_inside_anchors = keras_rcnn.backend.clip(all_inside_anchors, img_info[0][:2])

        inds_inside = keras.backend.eval(inds_inside)

        assert inds_inside.shape == (224,), keras.backend.eval(all_inside_anchors)

        all_inside_anchors = keras.backend.eval(all_inside_anchors)

        assert all_inside_anchors.shape == (224, 4)

    def test_inside_and_outside_weights_1(self, anchor_layer):
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

        output_inside_weights, output_outside_weights = anchor_layer._inside_and_outside_weights(anchors, subsample, positive_weight, proposed_inside_weights)

        target_inside_weights = keras.backend.eval(target_inside_weights)
        output_inside_weights = keras.backend.eval(output_inside_weights)

        numpy.testing.assert_array_equal(target_inside_weights, output_inside_weights)

        target_outside_weights = keras.backend.eval(target_outside_weights)
        output_outside_weights = keras.backend.eval(output_outside_weights)

        numpy.testing.assert_array_equal(target_outside_weights, output_outside_weights)

    def test_inside_and_outside_weights_2(self, anchor_layer):
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

        output_inside_weights, output_outside_weights = anchor_layer._inside_and_outside_weights(anchors, subsample, positive_weight, proposed_inside_weights)

        target_inside_weights = keras.backend.eval(target_inside_weights)
        output_inside_weights = keras.backend.eval(output_inside_weights)

        numpy.testing.assert_array_equal(target_inside_weights, output_inside_weights)

        target_outside_weights = keras.backend.eval(target_outside_weights)
        output_outside_weights = keras.backend.eval(output_outside_weights)

        numpy.testing.assert_array_equal(target_outside_weights, output_outside_weights)
