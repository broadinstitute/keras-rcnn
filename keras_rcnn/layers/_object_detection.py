# -*- coding: utf-8 -*-

import keras.backend
import keras.engine.topology
import tensorflow

import keras_rcnn.backend


class ObjectDetection(keras.engine.topology.Layer):
    """
    Get final detections + labels by unscaling back to image space, applying
    regression deltas, choosing box coordinates, and removing extra
    detections via NMS

    Arguments:
        threshold: objects with maximum score less than threshold are thrown
        out

        test_nms: A float representing the threshold for deciding whether
        boxes overlap too much with respect to IoU

    """
    def __init__(self, maximum=300, threshold=0.5, **kwargs):
        self.maximum = maximum

        self.threshold = threshold

        super(ObjectDetection, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectDetection, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """
        # Inputs
        proposals: output of proposal target (1, N, 4)
        deltas: predicted deltas (1, N, 4*classes)
        scores: score distributions (1, N, classes)
        metadata: image information (1, 3)

        # Returns

        bounding_boxes: predicted boxes (1, N, 4 * classes)

        scores: score distribution over all classes (1, N, classes),
        note the box only corresponds to the most probable class, not the
        other classes
        """
        def detections(num_output):
            output_bounding_boxes, output_deltas, output_scores, target_metadata = inputs[0], inputs[1], inputs[2], inputs[3]

            output_bounding_boxes = keras.backend.reshape(output_bounding_boxes, (-1, 4))

            # unscale back to raw image space

            num_objects = keras.backend.shape(output_bounding_boxes)[0]

            output_bounding_boxes = output_bounding_boxes / target_metadata[0][2]

            output_deltas = keras.backend.reshape(output_deltas, (num_objects, -1))

            # Apply bounding-box regression deltas
            output_bounding_boxes = keras_rcnn.backend.bbox_transform_inv(output_bounding_boxes, output_deltas)

            output_bounding_boxes = keras_rcnn.backend.clip(output_bounding_boxes, target_metadata[0][:2])

            output_scores = keras.backend.reshape(output_scores, (num_objects, -1))

            # Arg max
            indicies = keras.backend.expand_dims(keras.backend.arange(0, num_objects, dtype='int64'))

            top_classes = keras.backend.expand_dims(keras.backend.argmax(output_scores, axis=1))

            coordinates = [
                keras.backend.concatenate([indicies, top_classes * 4], 1),
                keras.backend.concatenate([indicies, top_classes * 4 + 1], 1),
                keras.backend.concatenate([indicies, top_classes * 4 + 2], 1),
                keras.backend.concatenate([indicies, top_classes * 4 + 3], 1)
            ]

            coordinates = keras.backend.reshape(keras.backend.concatenate(coordinates, 1), (-1, 2))

            output_bounding_boxes = keras_rcnn.backend.gather_nd(output_bounding_boxes, coordinates)

            output_bounding_boxes = keras.backend.reshape(output_bounding_boxes, (-1, 4))

            output_scores = keras.backend.max(output_scores, axis=1)

            nms_indices = keras_rcnn.backend.non_maximum_suppression(output_bounding_boxes, output_scores, maximum=num_objects, threshold=self.threshold)

            output_bounding_boxes = keras.backend.gather(output_bounding_boxes, nms_indices)

            output_bounding_boxes = keras.backend.expand_dims(output_bounding_boxes, 0)

            output_bounding_boxes = self.pad(output_bounding_boxes)

            output_scores = keras.backend.gather(output_scores, nms_indices)

            output_scores = keras.backend.expand_dims(output_scores, 0)

            output_scores = self.pad(output_scores)

            return [output_bounding_boxes, output_scores][num_output]

        bounding_boxes = keras.backend.in_train_phase(inputs[0], lambda: detections(0), training=training)

        scores = keras.backend.in_train_phase(inputs[2], lambda: detections(1), training=training)

        return [bounding_boxes, scores]

    def compute_output_shape(self, input_shape):
        return [(1, input_shape[0][0], input_shape[1][2]), (1, input_shape[0][0], input_shape[2][2])]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    def pad(self, x):
        detections = keras.backend.shape(x)[1]

        pad_width = ((0, 0), (0, (keras.backend.max([0, (self.maximum - detections)]))), (0, 0))

        return keras_rcnn.backend.pad(x, pad_width, "constant")
