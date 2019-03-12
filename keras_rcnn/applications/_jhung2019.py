# -*- coding: utf-8 -*-

import keras
import numpy

import keras_rcnn.layers
import keras_rcnn.models.backbone

import keras_resnet
import keras_resnet.models


class JHung2019(keras.models.Model):
    """
    The region-based convolutional neural network (RCNN) used in
    Hung, et al. (2019).

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

    anchor_scales : An array-like with shape:

            $$(scales,)$$

        used to generate anchors. A scale corresponds to:

            $$area_{scale}=\sqrt{\frac{area_{anchor}}{area_{base}}}$$.

    anchor_stride : A positive integer

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
            anchor_padding=1,
            anchor_scales=None,
            anchor_stride=16,
            dense_units=1024,
            maximum_proposals=300,
            minimum_size=16
    ):
        if anchor_aspect_ratios is None:
            anchor_aspect_ratios = [0.5, 1.0, 2.0]

        if anchor_scales is None:
            anchor_scales = [32, 64, 128, 256, 512]

        self.n_categories = len(categories) + 1

        k = len(anchor_aspect_ratios)

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

        target_metadata = keras.layers.Input(
            shape=(3,),
            name="target_metadata"
        )

        inputs = [
            target_bounding_boxes,
            target_categories,
            target_image,
            target_metadata
        ]

        backbone = keras_resnet.models.ResNet2D50(target_image, freeze_bn=True)

        features = backbone.outputs

        convolution_3x3 = keras.layers.Conv2D(
            kernel_size=(3, 3),
            filters=64,
            name="3x3",
            padding='same'
        )(features)

        output_deltas = keras.layers.Conv2D(
            filters=k * 4,
            kernel_size=(1, 1),
            activation="linear",
            name="deltas1",
            padding='same'
        )(convolution_3x3)

        output_scores = keras.layers.Conv2D(
            filters=k * 2,
            kernel_size=(1, 1),
            activation="sigmoid",
            name="scores1",
            padding='valid'
        )(convolution_3x3)

        target_anchors, target_proposal_bounding_boxes, target_proposal_categories = keras_rcnn.layers.Anchor(
            base_size=minimum_size,
            padding=anchor_padding,
            aspect_ratios=anchor_aspect_ratios,
            scales=anchor_scales,
            stride=anchor_stride
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

        (
            target_proposal_bounding_boxes,
            target_proposal_categories,
            output_proposal_bounding_boxes
        ) = keras_rcnn.layers.ProposalTarget()([
            target_bounding_boxes,
            target_categories,
            output_proposal_bounding_boxes
        ])

        output_features = keras_rcnn.layers.RegionOfInterestAlignPyramid(
            extent=(7, 7),
            strides=1
        )([
            target_metadata,
            output_proposal_bounding_boxes,
            features
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
                name="fc2"
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

        output_bounding_boxes, output_categories, mask_features = keras_rcnn.layers.ObjectDetection()(
            [
                target_metadata,
                output_deltas,
                output_proposal_bounding_boxes,
                output_scores,
            ])

        outputs = [
            output_bounding_boxes,
            output_categories
        ]

        super(JHung2019, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(JHung2019, self).compile(optimizer, None)

        origin = "http://keras-rcnn.storage.googleapis.com/JHung2019.tar.gz"

        self.load_weights("JHung2019.hdf5", by_name=True)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        target_bounding_boxes = numpy.zeros((x.shape[0], 1, 4))

        target_categories = numpy.zeros((x.shape[0], 1, self.n_categories))

        target_metadata = numpy.array([[x.shape[1], x.shape[2], 1.0]])

        x = [
            target_bounding_boxes,
            target_categories,
            x,
            target_metadata
        ]

        return super(JHung2019, self).predict(x, batch_size, verbose, steps)
