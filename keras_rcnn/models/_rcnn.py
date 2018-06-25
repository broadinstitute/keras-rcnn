# -*- coding: utf-8 -*-

import keras
import numpy

import keras_rcnn.layers
import keras_rcnn.models.backbone


class RCNN(keras.models.Model):
    """
    A Region-based Convolutional Neural Network (RCNN)

    Parameters
    ----------

    input_shape : A shape tuple (integer) without the batch dimension.

        For example:

            `input_shape=(224, 224, 3)`

        specifies that the input are batches of $224 × 224$ RGB images.

        Likewise:

            `input_shape=(224, 224)`

        specifies that the input are batches of $224 × 224$ grayscale
        images.

    categories : An array-like with shape:

            $$(categories,)$$.

        For example:

            `categories=["circle", "square", "triangle"]`

        specifies that the detected objects belong to either the
        “circle,” “square,” or “triangle” category.

    anchor_aspect_ratios : An array-like with shape:

            $$(aspect_ratios,)$$

        used to generate anchors.

        For example:

            `aspect_ratios=[0.5, 1., 2.]`

        corresponds to 1:2, 1:1, and 2:1 respectively.

    anchor_base_size : Integer that specifies an anchor’s base area:

            $$base_area = base_size^{2}$$.

    anchor_scales : An array-like with shape:

            $$(scales,)$$

        used to generate anchors. A scale corresponds to:

            $$area_{scale}=\sqrt{\frac{area_{anchor}}{area_{base}}}$$.

    anchor_stride : A positive integer

    backbone :

    dense_units : A positive integer that specifies the dimensionality of
        the fully-connected layers.

        The fully-connected layers are the layers that precede the
        fully-connected layers for the classification, regression and
        segmentation target functions.

        Increasing the number of dense units will increase the
        expressiveness of the network and consequently the ability to
        correctly learn the target functions, but it’ll substantially
        increase the number of learnable parameters and memory needed by
        the model.

    extent : A shape tuple (integer).

    mask_shape : A shape tuple (integer).

    maximum_proposals : A positive integer that specifies the maximum
        number of object proposals returned from the model.

        The model always return an array-like with shape:

            $$(maximum_proposals, 4)$$

        regardless of the number of object proposals returned after
        non-maximum suppression is performed. If the number of object
        proposals returned from non-maximum suppression is less than the
        number of objects specified by the `maximum_proposals` parameter,
        the model will return bounding boxes with the value:

            `[0., 0., 0., 0.]`

        and scores with the value `[0.]`.

    minimum_size : A positive integer that specifies the maximum width
        or height for each object proposal.
    """
    def __init__(
            self,
            input_shape,
            categories,
            anchor_aspect_ratios=None,
            anchor_base_size=16,
            anchor_padding=1,
            anchor_scales=None,
            anchor_stride=16,
            backbone=None,
            dense_units=1024,
            extent=(14, 14),
            mask_shape=(28, 28),
            maximum_proposals=300,
            minimum_size=16
    ):
        if anchor_aspect_ratios is None:
            anchor_aspect_ratios = [0.5, 1.0, 2.0]

        if anchor_scales is None:
            anchor_scales = [4, 8, 16]

        self.mask_shape = mask_shape

        self.n_categories = len(categories) + 1

        k = len(anchor_aspect_ratios) * len(anchor_scales)

        target_bounding_boxes = keras.layers.Input(
            shape=(None, 4),
            name="target_bounding_boxes"
        )

        target_categories = keras.layers.Input(
            shape=(None, self.n_categories),
            name="target_categories"
        )

        target_image = keras.layers.Input(
            shape=input_shape,
            name="target_image"
        )

        target_masks = keras.layers.Input(
            shape=(None,) + mask_shape,
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
            convolution_2, convolution_3, convolution_4, convolution_5 = backbone()(target_image)
        else:
            convolution_2, convolution_3, convolution_4, convolution_5 = keras_rcnn.models.backbone.ResNet50()(target_image)

        # Feature Pyramid Network (FPN)

        pyramid_5 = keras.layers.Conv2D(256, (1, 1))(convolution_5)

        pyramid_4 = keras.layers.Add()(
            [
                keras.layers.UpSampling2D((2, 2))(pyramid_5),
                keras.layers.Conv2D(256, (1, 1))(convolution_4)
            ]
        )

        pyramid_3 = keras.layers.Add()(
            [
                keras.layers.UpSampling2D((2, 2))(pyramid_4),
                keras.layers.Conv2D(256, (1, 1))(convolution_3)
            ]
        )

        pyramid_2 = keras.layers.Add()(
            [
                keras.layers.UpSampling2D((2, 2))(pyramid_3),
                keras.layers.Conv2D(256, (1, 1))(convolution_2)
            ]
        )

        pyramid_2 = keras.layers.Conv2D(256, (3, 3), padding="same")(pyramid_2)
        pyramid_3 = keras.layers.Conv2D(256, (3, 3), padding="same")(pyramid_3)
        pyramid_4 = keras.layers.Conv2D(256, (3, 3), padding="same")(pyramid_4)
        pyramid_5 = keras.layers.Conv2D(256, (3, 3), padding="same")(pyramid_5)

        pyramid_6 = keras.layers.MaxPooling2D((1, 1), strides=2)(pyramid_5)

        proposal_features = [
            pyramid_2,
            pyramid_3,
            pyramid_4,
            pyramid_5,
            pyramid_6
        ]

        output_features = [
            pyramid_2,
            pyramid_3,
            pyramid_4,
            pyramid_5
        ]

        # Region Proposal Network (RPN)

        proposals = []

        for features in proposal_features:
            output_deltas = keras.layers.Conv2D(
                filters=k * 4,
                kernel_size=(1, 1),
                activation="linear",
                kernel_initializer="zero"
            )(features)

            output_scores = keras.layers.Conv2D(
                filters=k * 1,
                kernel_size=(1, 1),
                activation="sigmoid",
                kernel_initializer="uniform"
            )(features)

            proposals += [output_deltas, output_scores]

        proposals = list(zip(*proposals))

        outputs = [keras.layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(proposals, ["output_deltas", "output_scores"])]

        output_deltas, output_scores = outputs

        target_anchors, target_proposal_bounding_boxes, target_proposal_categories = keras_rcnn.layers.Anchor(
            padding=anchor_padding,
            aspect_ratios=anchor_aspect_ratios,
            base_size=anchor_base_size,
            scales=anchor_scales,
            stride=anchor_stride,
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

        output_proposal_bounding_boxes = keras_rcnn.layers.ObjectProposal(
            maximum_proposals=maximum_proposals,
            minimum_size=minimum_size
        )([
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

        mask_features = self._mask_network()(
            [
                target_metadata,
                output_features,
                output_proposal_bounding_boxes
            ]
        )

        output_features = keras_rcnn.layers.RegionOfInterest(
            extent=extent,
            strides=1
        )([
            target_metadata,
            output_features,
            output_proposal_bounding_boxes
        ])

        output_features = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=dense_units,
                activation="relu",
                name="fc1"
            )
        )(output_features)

        output_features = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=dense_units,
                activation="relu",
                name="fc1"
            )
        )(output_features)

        output_features = keras.layers.TimeDistributed(
            keras.layers.Flatten()
        )(output_features)

        output_deltas = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=4 * self.n_categories,
                activation="linear",
                kernel_initializer="zero",
                name="deltas2"
            )
        )(output_features)

        output_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * self.n_categories,
                activation="softmax",
                kernel_initializer="zero",
                name="scores2"
            )
        )(output_features)

        output_deltas, output_scores = keras_rcnn.layers.RCNN()([
            target_proposal_bounding_boxes,
            target_proposal_categories,
            output_deltas,
            output_scores
        ])

        output_bounding_boxes, output_categories = keras_rcnn.layers.ObjectDetection()([
            target_metadata,
            output_deltas,
            output_proposal_bounding_boxes,
            output_scores
        ])

        outputs = [
            output_bounding_boxes,
            output_categories
        ]

        super(RCNN, self).__init__(inputs, outputs)

    def _mask_network(self):
        def f(x):
            target_metadata, output_features, output_proposal_bounding_boxes = x

            mask_features = keras_rcnn.layers.RegionOfInterest(
                extent=(14, 14),
                strides=2
            )([
                target_metadata,
                output_features,
                output_proposal_bounding_boxes
            ])

            mask_features = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    activation="relu",
                    filters=256,
                    kernel_size=(3, 3),
                    padding="same"
                )
            )(mask_features)

            mask_features = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    activation="relu",
                    filters=256,
                    kernel_size=(3, 3),
                    padding="same"
                )
            )(mask_features)

            mask_features = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    activation="relu",
                    filters=256,
                    kernel_size=(3, 3),
                    padding="same"
                )
            )(mask_features)

            mask_features = keras.layers.TimeDistributed(
                keras.layers.Conv2DTranspose(
                    activation="relu",
                    filters=256,
                    kernel_size=(2, 2),
                    strides=2
                )
            )(mask_features)

            mask_features = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    activation="sigmoid",
                    filters=self.n_categories,
                    kernel_size=(1, 1),
                    strides=1
                )
            )(mask_features)

            return mask_features

        return f

    def compile(self, optimizer, **kwargs):
        super(RCNN, self).compile(optimizer, None)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        target_bounding_boxes = numpy.zeros((x.shape[0], 1, 4))

        target_categories = numpy.zeros((x.shape[0], 1, self.n_categories))

        target_mask = numpy.zeros((1, 1, *self.mask_shape))

        target_metadata = numpy.array([[x.shape[1], x.shape[2], 1.0]])

        x = [
            target_bounding_boxes,
            target_categories,
            x,
            target_mask,
            target_metadata
        ]

        return super(RCNN, self).predict(x, batch_size, verbose, steps)
