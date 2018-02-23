# -*- coding: utf-8 -*-

import keras

import keras_rcnn.layers
import keras_rcnn.models.backbone


class RCNN(keras.models.Model):
    def __init__(self, input_shape, categories, backbone=None, **kwargs):
        n_categories = len(categories) + 1

        aspect_ratios = [0.5, 1.0, 2.0]

        scales = [4, 8, 16]

        if "anchor_target" in kwargs:
            if "aspect_ratios" in kwargs["anchor_target"]:
                aspect_ratios = kwargs["anchor_target"]["aspect_ratios"]

            if "scales" in kwargs["anchor_target"]:
                scales = kwargs["anchor_target"]["scales"]

        k = len(aspect_ratios) * len(scales)

        target_bounding_boxes = keras.layers.Input(
            shape=(None, 4),
            name="target_bounding_boxes"
        )

        target_categories = keras.layers.Input(
            shape=(None, n_categories),
            name="target_categories"
        )

        target_image = keras.layers.Input(
            shape=input_shape,
            name="target_image"
        )

        target_masks = keras.layers.Input(
            shape=(None, 28, 28),
            name="target_masks"
        )

        target_metadata = keras.layers.Input(
            shape=(3,),
            name="target_metadata"
        )

        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        inputs = [
            target_bounding_boxes,
            target_categories,
            target_image,
            target_masks,
            target_metadata
        ]

        if backbone:
            output_features = backbone()(target_image)
        else:
            output_features = keras_rcnn.models.backbone.VGG16()(target_image)

        convolution_3x3 = keras.layers.Conv2D(64, **options)(output_features)

        output_deltas = keras.layers.Conv2D(
            filters=k * 4,
            kernel_size=(1, 1),
            activation="linear",
            kernel_initializer="zero",
            name="deltas"
        )(convolution_3x3)

        output_scores = keras.layers.Conv2D(
            filters=k * 1,
            kernel_size=(1, 1),
            activation="sigmoid",
            kernel_initializer="uniform",
            name="scores"
        )(convolution_3x3)

        target_anchors, target_proposal_bounding_boxes, target_proposal_categories = keras_rcnn.layers.AnchorTarget(
            aspect_ratios=aspect_ratios,
            scales=scales
        )([
            target_bounding_boxes,
            target_metadata,
            output_scores
        ])

        output_deltas, output_scores = keras_rcnn.layers.RPN()([
            target_proposal_bounding_boxes,
            target_proposal_categories,
            output_deltas,
            output_scores
        ])

        output_proposal_bounding_boxes = keras_rcnn.layers.ObjectProposal()([
            target_anchors,
            target_metadata,
            output_deltas,
            output_scores
        ])

        target_proposal_bounding_boxes, target_proposal_categories, output_proposal_bounding_boxes = keras_rcnn.layers.ProposalTarget()([
            target_bounding_boxes,
            target_categories,
            output_proposal_bounding_boxes
        ])

        output_features = keras_rcnn.layers.RegionOfInterest((14, 14))([
            target_metadata,
            output_features,
            output_proposal_bounding_boxes
        ])

        output_features = keras.layers.TimeDistributed(
            keras.layers.Flatten()
        )(output_features)

        output_features = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=512,
                activation="relu"
            )
        )(output_features)

        output_deltas = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=4 * n_categories,
                activation="linear",
                kernel_initializer="zero"
            )
        )(output_features)

        output_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * n_categories,
                activation="softmax",
                kernel_initializer="zero"
            )
        )(output_features)

        output_deltas, output_scores = keras_rcnn.layers.RCNN()([
            target_proposal_bounding_boxes,
            target_proposal_categories,
            output_deltas,
            output_scores
        ])

        output_bounding_boxes, output_categories = keras_rcnn.layers.ObjectDetection()([
            output_proposal_bounding_boxes,
            output_deltas,
            output_scores,
            target_metadata
        ])

        outputs = [
            output_bounding_boxes,
            output_categories
        ]

        super(RCNN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RCNN, self).compile(optimizer, None)
