# -*- coding: utf-8 -*-

import keras.backend
import keras.engine.topology
import tensorflow

import keras_rcnn.backend


class ObjectDetection(keras.engine.topology.Layer):
    def __init__(self, padding=300, **kwargs):
        self.padding = padding

        super(ObjectDetection, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectDetection, self).build(input_shape)

    def call(self, x, training=None, **kwargs):
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
            proposals, deltas, scores, metadata = x[0], x[1], x[2], x[3]

            proposals = keras.backend.reshape(proposals, (-1, 4))

            # unscale back to raw image space

            boxes = proposals / metadata[0][2]
            num_objects = keras.backend.shape(proposals)[0]
            deltas = keras.backend.reshape(deltas, (num_objects, -1))

            # Apply bounding-box regression deltas
            pred_boxes = keras_rcnn.backend.bbox_transform_inv(boxes, deltas)

            pred_boxes = keras_rcnn.backend.clip(pred_boxes, metadata[0][:2])

            scores = keras.backend.reshape(scores, (num_objects, -1))

            # Arg max
            inds = keras.backend.expand_dims(keras.backend.arange(0, num_objects, dtype="int64"))

            top_classes = keras.backend.expand_dims(keras.backend.argmax(scores, axis=1))

            coordinate_0 = keras.backend.concatenate([inds, top_classes * 4], 1)
            coordinate_1 = keras.backend.concatenate([inds, top_classes * 4 + 1], 1)
            coordinate_2 = keras.backend.concatenate([inds, top_classes * 4 + 2], 1)
            coordinate_3 = keras.backend.concatenate([inds, top_classes * 4 + 3], 1)

            pred_boxes = keras_rcnn.backend.gather_nd(pred_boxes, keras.backend.reshape(keras.backend.concatenate([coordinate_0, coordinate_1, coordinate_2, coordinate_3], 1), (-1, 2)))

            pred_boxes = keras.backend.reshape(pred_boxes, (-1, 4))

            max_scores = keras.backend.max(scores[:, 1:], axis=1)

            nms_indices = keras_rcnn.backend.non_maximum_suppression(boxes=pred_boxes, scores=max_scores, maximum=num_objects, threshold=0.5)

            pred_boxes = keras.backend.gather(pred_boxes, nms_indices)

            scores = keras.backend.gather(scores, nms_indices)

            pred_boxes = keras.backend.expand_dims(pred_boxes, 0)

            scores = keras.backend.expand_dims(scores, 0)

            pred_boxes = self.pad(pred_boxes, self.padding)

            scores = self.pad(scores, self.padding)

            detections = [pred_boxes, scores]

            return detections[num_output]

        bounding_boxes = keras.backend.in_train_phase(x[0], lambda: detections(0), training=training)

        scores = keras.backend.in_train_phase(x[2], lambda: detections(1), training=training)

        return [bounding_boxes, scores]

    def compute_output_shape(self, input_shape):
        return [(1, input_shape[0][0], input_shape[1][2]), (1, input_shape[0][0], input_shape[2][2])]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    @staticmethod
    def pad(x, padding):
        detections = keras.backend.shape(x)[1]

        difference = padding - detections

        difference = keras.backend.max([0, difference])

        paddings = ((0, 0), (0, difference), (0, 0))

        return tensorflow.pad(x, paddings, mode="constant")
