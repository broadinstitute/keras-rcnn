# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import keras.layers

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.preprocessing


def _extract_features(options=None, training=None):
    if options is None:
        options = {}

    def f(inputs):
        features = keras.layers.Conv2D(64, name="convolution_1_1", **options)(inputs)
        features = keras.layers.Conv2D(64, name="convolution_1_2", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_1")(features)

        features = keras.layers.Conv2D(128, name="convolution_2_1", **options)(features)
        features = keras.layers.Conv2D(128, name="convolution_2_2", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_2")(features)

        features = keras.layers.Conv2D(256, name="convolution_3_1", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_2", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_3", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_4", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_3")(features)

        features = keras.layers.Conv2D(512, name="convolution_4_1", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_2", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_3", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_4", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_4")(features)

        features = keras.layers.Conv2D(512, name="convolution_5_1", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_5_2", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_5_3", **options)(features)

        return features

    return f


def _extract_proposals(options=None):
    if options is None:
        options = {}

    def f(inputs):
        convolution_3x3 = keras.layers.Conv2D(512, name="convolution_3x3", **options)(inputs)

        deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)
        scores = keras.layers.Conv2D(9 * 2, (1, 1), kernel_initializer="uniform", name="scores")(convolution_3x3)

        return [deltas, scores]

    return f


def _extract_regions(classes):
    def f(inputs):
        features, metadata, proposals = inputs

        regions = keras_rcnn.layers.RegionOfInterest(extent=(14, 14))([features, proposals, metadata])

        regions = keras.layers.TimeDistributed(keras.layers.Flatten())(regions)

        regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(regions)
        regions = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(regions)

        regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(regions)
        regions = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(regions)

        deltas = keras.layers.TimeDistributed(keras.layers.Dense(4 * classes, activation="linear", kernel_initializer="zero"))(regions)
        scores = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax", kernel_initializer="zero"))(regions)

        return [deltas, scores]

    return f


def _train(classes, training_options=None):
    if training_options is None:
        training_options = {
            "anchor_target": {
                "allowed_border": 0,
                "clobber_positives": False,
                "negative_overlap": 0.3,
                "positive_overlap": 0.7,
            },
            "object_proposal": {
                "maximum_proposals": 300,
                "minimum_size": 16,
                "stride": 16
            },
            "object_detection": {

            },
            "proposal_target": {
                "fg_fraction": 0.5,
                "fg_thresh": 0.7,
                "bg_thresh_hi": 0.5,
                "bg_thresh_lo": 0.1,
            }
        }

    def f(inputs):
        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        bounding_boxes, image, labels, metadata = inputs

        features = _extract_features(options)(image)

        deltas, scores = _extract_proposals(options)(features)

        all_anchors, rpn_labels, bounding_box_targets = keras_rcnn.layers.AnchorTarget(
            allowed_border=training_options["anchor_target"]["allowed_border"],
            clobber_positives=training_options["anchor_target"]["clobber_positives"],
            negative_overlap=training_options["anchor_target"]["negative_overlap"],
            positive_overlap=training_options["anchor_target"]["positive_overlap"]
        )([scores, bounding_boxes, metadata])

        scores = keras.layers.Activation("softmax")(scores)

        deltas = keras_rcnn.layers.RPNRegressionLoss(9)([deltas, bounding_box_targets, rpn_labels])
        scores = keras_rcnn.layers.RPNClassificationLoss(9)([scores, rpn_labels])

        proposals_ = keras_rcnn.layers.ObjectProposal(
            maximum_proposals=training_options["object_proposal"]["maximum_proposals"],
            minimum_size=training_options["object_proposal"]["minimum_size"],
            stride=training_options["object_proposal"]["stride"]
        )([metadata, deltas, scores, all_anchors])

        proposals, labels_targets, bounding_box_targets = keras_rcnn.layers.ProposalTarget(
            fg_fraction=training_options["proposal_target"]["fg_fraction"],
            fg_thresh=training_options["proposal_target"]["fg_thresh"],
            bg_thresh_hi=training_options["proposal_target"]["bg_thresh_hi"],
            bg_thresh_lo=training_options["proposal_target"]["bg_thresh_lo"]
        )([proposals_, labels, bounding_boxes])

        deltas, scores = _extract_regions(classes)([features, metadata, proposals])

        deltas = keras_rcnn.layers.losses.RCNNRegressionLoss()([deltas, bounding_box_targets, labels_targets])
        scores = keras_rcnn.layers.losses.RCNNClassificationLoss()([scores, labels_targets])

        bounding_boxes, scores = keras_rcnn.layers.ObjectDetection()([proposals, deltas, scores, metadata])

        return [bounding_boxes, proposals, scores]

    return f


class RCNN(keras.models.Model):
    def __init__(self, image, classes, training_options=None):
        inputs = [
            keras.layers.Input((None, 4)),
            image,
            keras.layers.Input((None, classes)),
            keras.layers.Input((3,))
        ]

        outputs = _train(classes, training_options)(inputs)

        super(RCNN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RCNN, self).compile(optimizer, None)
