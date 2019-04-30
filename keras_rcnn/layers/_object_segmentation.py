# -*- coding: utf-8 -*-

import keras.backend
import keras.engine.topology
import tensorflow

import keras_rcnn.backend


class ObjectSegmentation(keras.layers.Layer):
    def __init__(self, padding=300, **kwargs):
        self.padding = padding

        super(ObjectSegmentation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectSegmentation, self).build(input_shape)

    def call(self, x, training=None, **kwargs):
        """
        # Inputs
        metadata: image information (1, 3)
        deltas: predicted deltas (1, N, 4*classes)
        proposals: output of proposal target (1, N, 4)
        scores: score distributions (1, N, classes)

        # Returns
        bounding_boxes: predicted boxes (1, N, 4 * classes)
        scores: score distribution over all classes (1, N, classes),
        note the box only corresponds to the most probable class, not the
        other classes
        """

        metadata, deltas, proposals, scores, masks = x[0], x[1], x[2], x[3], x[4]

        bounding_boxes = keras.backend.in_train_phase(
            proposals,
            lambda: self.detections(
                0,
                metadata,
                deltas,
                proposals,
                scores,
                masks
            ),
            training=training
        )

        masks = keras.backend.in_train_phase(
            masks,
            lambda: self.detections(
                2,
                metadata,
                deltas,
                proposals,
                scores,
                masks
            ),
            training=training
        )

        scores = keras.backend.in_train_phase(
            scores,
            lambda: self.detections(
                1,
                metadata,
                deltas,
                proposals,
                scores,
                masks
            ),
            training=training
        )

        return [bounding_boxes, scores, masks]

    def compute_output_shape(self, input_shape):
        return [
            (1, input_shape[0][0], input_shape[1][2]),
            (1, input_shape[0][0], input_shape[2][2]),
            (1, input_shape[0][0], input_shape[2][2], input_shape[4][2], input_shape[4][3], input_shape[4][4])
        ]

    def compute_mask(self, inputs, mask=None):
        return 3 * [None]

    def detections(self, num_output, metadata, deltas, proposals, scores, masks):
        proposals = keras.backend.reshape(proposals, (-1, 4))

        # unscale back to raw image space
        bounding_boxes = proposals / metadata[0][2]

        num_objects = keras.backend.shape(proposals)[0]

        deltas = keras.backend.reshape(deltas, (num_objects, -1))

        # Apply bounding-box regression deltas
        predicted_bounding_boxes = keras_rcnn.backend.bbox_transform_inv(
            bounding_boxes,
            deltas
        )

        predicted_bounding_boxes = keras_rcnn.backend.clip(
            predicted_bounding_boxes,
            metadata[0][:2]
        )

        scores = keras.backend.reshape(scores, (num_objects, -1))

        # Arg max
        inds = keras.backend.expand_dims(
            keras.backend.arange(0, num_objects, dtype="int64")
        )

        top_classes = keras.backend.expand_dims(
            keras.backend.argmax(scores, axis=1)
        )

        coordinate_0 = keras.backend.concatenate(
            [inds, top_classes * 4],
            1
        )

        coordinate_1 = keras.backend.concatenate(
            [inds, top_classes * 4 + 1],
            1
        )

        coordinate_2 = keras.backend.concatenate(
            [inds, top_classes * 4 + 2],
            1
        )

        coordinate_3 = keras.backend.concatenate(
            [inds, top_classes * 4 + 3],
            1
        )

        predicted_bounding_boxes = keras_rcnn.backend.gather_nd(
            predicted_bounding_boxes,
            keras.backend.reshape(
                keras.backend.concatenate(
                    [
                        coordinate_0,
                        coordinate_1,
                        coordinate_2,
                        coordinate_3
                    ],
                    1
                ),
                (-1, 2)
            )
        )

        predicted_bounding_boxes = keras.backend.reshape(
            predicted_bounding_boxes,
            (-1, 4)
        )

        max_scores = keras.backend.max(scores[:, 1:], axis=1)

        suppressed_indices = keras_rcnn.backend.non_maximum_suppression(
            boxes=predicted_bounding_boxes,
            scores=max_scores,
            maximum=num_objects,
            threshold=0.5
        )

        predicted_bounding_boxes = keras.backend.gather(
            predicted_bounding_boxes,
            suppressed_indices
        )

        scores = keras.backend.gather(scores, suppressed_indices)

        predicted_bounding_boxes = keras.backend.expand_dims(
            predicted_bounding_boxes,
            0
        )

        scores = keras.backend.expand_dims(scores, 0)

        predicted_bounding_boxes = self.pad_bounding_boxes(
            predicted_bounding_boxes,
            self.padding
        )

        scores = self.pad_bounding_boxes(scores, self.padding)

        masks = keras.backend.squeeze(masks, axis=0)

        masks = keras.backend.gather(masks, suppressed_indices)

        masks = keras.backend.expand_dims(masks, axis=0)

        masks = self.pad_masks(masks, self.padding)

        detections = [predicted_bounding_boxes, scores, masks]

        return detections[num_output]

    @staticmethod
    def pad_bounding_boxes(x, padding):
        detections = keras.backend.shape(x)[1]

        difference = padding - detections

        difference = keras.backend.max([0, difference])

        paddings = ((0, 0), (0, difference), (0, 0))

        # TODO: replace with `keras.backend.pad`
        return tensorflow.pad(x, paddings, mode="constant")

    @staticmethod
    def pad_masks(x, padding):
        detections = keras.backend.shape(x)[1]

        difference = padding - detections

        difference = keras.backend.max([0, difference])

        paddings = ((0, 0), (0, difference), (0, 0), (0, 0), (0, 0))

        # TODO: replace with `keras.backend.pad`
        return tensorflow.pad(x, paddings, mode="constant")

    def get_config(self):
        configuration = {
            "padding": self.padding
        }

        return {
            **super(ObjectSegmentation, self).get_config(), **configuration
        }
