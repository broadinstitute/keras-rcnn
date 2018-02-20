# -*- coding: utf-8 -*-

import keras
import keras_resnet.models

import keras_rcnn.layers


class RPN(keras.models.Model):
    def __init__(self, input_shape, categories):
        n_categories = len(categories) + 1

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

        _, _, _, output_features = keras_resnet.models.ResNet50(
            inputs=target_image,
            include_top=False
        ).outputs

        convolution_3x3 = keras.layers.Conv2D(64, **options)(output_features)

        output_deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)

        output_scores = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="scores")(convolution_3x3)

        target_anchors, target_proposal_bounding_boxes, target_proposal_categories = keras_rcnn.layers.AnchorTarget()([
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

        target_proposal_bounding_boxes, target_proposal_categories, _ = keras_rcnn.layers.ProposalTarget()([
            target_bounding_boxes,
            target_categories,
            output_proposal_bounding_boxes
        ])

        outputs = [
            target_anchors,
            target_proposal_bounding_boxes,
            target_proposal_categories,
            output_deltas,
            output_scores
        ]

        super(RPN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RPN, self).compile(optimizer, None)
