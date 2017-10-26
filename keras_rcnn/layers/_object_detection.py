# -*- coding: utf-8 -*-

import keras.backend
import keras.engine.topology

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

    def __init__(self, **kwargs):
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
        def boxes():
            proposals, deltas, _, metadata = x[0], x[1], x[2], x[3]

            proposals = keras.backend.reshape(proposals, (-1, 4))

            # unscale back to raw image space
            boxes = proposals / metadata[0][2]
            
            deltas = keras.backend.reshape(deltas, (keras.backend.shape(proposals)[0], -1))

            # Apply bounding-box regression deltas
            pred_boxes = keras_rcnn.backend.bbox_transform_inv(boxes, deltas)

            pred_boxes = keras_rcnn.backend.clip(pred_boxes, metadata[0][:2])

            return keras.backend.expand_dims(pred_boxes, 0)
        
        bounding_boxes = keras.backend.in_train_phase(x[1], lambda: boxes(), training=training)

        scores = x[2]

        return [bounding_boxes, scores]

    def compute_output_shape(self, input_shape):
        return [(1, input_shape[0][0], input_shape[1][2]), (1, input_shape[0][0], input_shape[2][2])]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]
