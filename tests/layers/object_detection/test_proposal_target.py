import keras.backend
import keras.utils
import numpy

import keras_rcnn.layers


class TestProposalTarget:
    def test_call(self):
        proposal_target = keras_rcnn.layers.ProposalTarget()

        proposals = numpy.random.random((1, 300, 4))
        proposals = keras.backend.variable(proposals)

        bounding_boxes = numpy.random.choice(range(0, 224), (1, 10, 4))
        bounding_boxes = keras.backend.variable(bounding_boxes)

        labels = numpy.random.choice(range(0, 2), (1, 10))
        labels = keras.utils.to_categorical(labels)
        labels = keras.backend.variable(labels)
        labels = keras.backend.expand_dims(labels, 0)

        proposal_target.call([proposals, bounding_boxes, labels])


def test_get_bbox_regression_labels():
    N = 10
    bbox_target_data = keras.backend.zeros((N, 4))
    num_classes = 3
    labels = numpy.reshape([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], (1, -1, 3))
    labels = keras.backend.variable(labels)
    bbox_targets = keras_rcnn.layers.object_detection._proposal_target.get_bbox_regression_labels(labels, bbox_target_data)
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
    rois, labels, bbox_targets = keras_rcnn.layers.object_detection._proposal_target.sample_rois(all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image, fg_thresh, bg_thresh_hi, bg_thresh_lo)
    assert keras.backend.eval(labels).shape == (N_proposals, num_classes)
    assert keras.backend.eval(rois).shape == (N_proposals, 4)
    assert keras.backend.eval(bbox_targets).shape == (N_proposals, 4*num_classes)