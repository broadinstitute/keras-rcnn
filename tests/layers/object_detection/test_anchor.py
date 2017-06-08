import keras.backend

import keras_rcnn.backend
import keras_rcnn.layers


class TestAnchor:
    def test_build(self):
        assert True

    def test_call(self, anchor_layer, gt_boxes):
        bbox_labels, bbox_reg_targets, inds_inside, n_all_bbox = anchor_layer.call(gt_boxes)

        assert bbox_labels.shape == (84, )

        assert keras.backend.int_shape(bbox_reg_targets) == (84, 4)

        assert inds_inside.shape == (84, )

        assert n_all_bbox == 1764

    # def test_predict(self):
    #     shape = (None, 5)
    #
    #     x = keras.layers.Input(shape)
    #
    #     y = keras_rcnn.layers.object_detection.Anchor((14, 14), (224, 224))(x)

    def test_regression(self, anchor_layer, gt_boxes):
        _, y_true, inds_inside, _ = anchor_layer.call(gt_boxes)

        y_true = keras.backend.eval(y_true)

        # anchor_layer.regression(y_true, y_pred, inds_inside)

        assert True

    def test_compute_output_shape(self, anchor_layer):
        assert anchor_layer.compute_output_shape((14, 14)) == (None, None, 4)

    def test_label(self, anchor_layer, feat_h, feat_w, gt_boxes, img_info):
        all_bbox = keras_rcnn.backend.shift((feat_h, feat_w), anchor_layer.stride)

        inds_inside, all_inside_bbox = anchor_layer._inside_image(all_bbox, img_info)

        argmax_overlaps_inds, bbox_labels = anchor_layer._label(gt_boxes, all_inside_bbox, inds_inside)

        assert argmax_overlaps_inds.shape == (84, )

        assert bbox_labels.shape == (84, )

    def test_inside_image(self, anchor_layer, img_info):
        all_anchors = anchor_layer.shifted_anchors

        inds_inside, all_inside_anchors = anchor_layer._inside_image(all_anchors, img_info)

        assert inds_inside.shape == (84,)

        assert all_inside_anchors.shape == (84, 4)

    def test_overlapping(self, anchor_layer, gt_boxes, img_info):
        all_anchors = anchor_layer.shifted_anchors

        inds_inside, all_inside_anchors = anchor_layer._inside_image(all_anchors, img_info)

        argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = anchor_layer._overlapping(gt_boxes, all_inside_anchors, inds_inside)

        assert argmax_overlaps_inds.shape == (84, )

        assert max_overlaps.shape == (84, )

        assert gt_argmax_overlaps_inds.shape == (91, )
