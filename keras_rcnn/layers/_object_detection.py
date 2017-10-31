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
        rois: output of proposal target (1, N, 4)
        deltas: predicted deltas (1, N, 4*classes)
        scores: score distributions (1, N, classes)
        metadata: image information (1, 3)

        # Returns
        pred_boxes: predicted boxes (1, N, 4 * classes)
        scores: score distribution over all classes (1, N, classes), note
        the box only corresponds to the most
            probable class, not the other classes
        """
        def boxes():
            rois, deltas, scores, metadata = x[0], x[1], x[2], x[3]

            rois = keras.backend.reshape(rois, (-1, 4))

            # unscale back to raw image space
            boxes = rois / metadata[0][2]     
            num_objects = keras.backend.shape(rois)[0]
            deltas = keras.backend.reshape(deltas, (num_objects, -1))

            # Apply bounding-box regression deltas
            pred_boxes = keras_rcnn.backend.bbox_transform_inv(boxes, deltas)

            pred_boxes = keras_rcnn.backend.clip(pred_boxes, metadata[0][:2])

            scores = keras.backend.reshape(scores, (num_objects, -1))

            # Arg max
            inds = keras.backend.expand_dims(keras.backend.arange(0, num_objects, dtype='int64'))
            top_classes = keras.backend.expand_dims(keras.backend.argmax(scores, axis=1))
            coordinate_0 = keras.backend.concatenate([inds, top_classes * 4], 1)
            coordinate_1 = keras.backend.concatenate([inds, top_classes * 4 + 1], 1)
            coordinate_2 = keras.backend.concatenate([inds, top_classes * 4 + 2], 1)
            coordinate_3 = keras.backend.concatenate([inds, top_classes * 4 + 3], 1)

            pred_boxes = keras_rcnn.backend.gather_nd(pred_boxes,
                                         keras.backend.reshape(
                                             keras.backend.concatenate([
                                                 coordinate_0,
                                                 coordinate_1,
                                                 coordinate_2,
                                                 coordinate_3
                                             ], 1),
                                             (-1, 2)
                                         ))
            pred_boxes = keras.backend.reshape(pred_boxes, (-1, 4))

            max_scores = keras.backend.max(scores, axis=1)

            nms_indices = keras_rcnn.backend.non_maximum_suppression(boxes=pred_boxes, scores=max_scores, maximum=num_objects, threshold=0.7)

            pred_boxes = keras.backend.gather(pred_boxes, nms_indices)

            return keras.backend.expand_dims(pred_boxes, 0)
        
        pred_boxes = keras.backend.in_train_phase(x[1], lambda: boxes(), training=training)

        return [pred_boxes, x[2]]

    def compute_output_shape(self, input_shape):
        return [(1, input_shape[0][0], input_shape[1][2]), (1, input_shape[0][0], input_shape[2][2])]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]
