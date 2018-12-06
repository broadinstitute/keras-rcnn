# -*- coding: utf-8 -*-

import keras.backend
import keras.engine.topology
import tensorflow

import keras_rcnn.backend


class ObjectDetection(keras.layers.Layer):
    def __init__(self, padding=300, **kwargs):
        self.padding = padding

        super(ObjectDetection, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectDetection, self).build(input_shape)

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

        def detections(num_output, metadata, deltas, proposals, scores, masks):
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

            pred_boxes = keras_rcnn.backend.gather_nd(pred_boxes, keras.backend.reshape(
                keras.backend.concatenate([coordinate_0, coordinate_1, coordinate_2, coordinate_3], 1), (-1, 2)))

            pred_boxes = keras.backend.reshape(pred_boxes, (-1, 4))

            max_scores = keras.backend.max(scores[:, 1:], axis=1)

            nms_indices = keras_rcnn.backend.non_maximum_suppression(boxes=pred_boxes, scores=max_scores,
                                                                     maximum=num_objects, threshold=0.5)

            pred_boxes = keras.backend.gather(pred_boxes, nms_indices)

            scores = keras.backend.gather(scores, nms_indices)

            pred_boxes = keras.backend.expand_dims(pred_boxes, 0)

            scores = keras.backend.expand_dims(scores, 0)

            pred_boxes = self.pad(pred_boxes, self.padding)

            scores = self.pad(scores, self.padding)

            masks = keras.backend.squeeze(masks, axis=0)

            masks = keras.backend.gather(masks, nms_indices)

            masks = keras.backend.expand_dims(masks, axis=0)

            masks = self.padmasks(masks, self.padding)


            detections = [pred_boxes, scores, masks]

            return detections[num_output]

        bounding_boxes = keras.backend.in_train_phase(proposals,
                                                      lambda: detections(0, metadata, deltas, proposals, scores, masks),
                                                      training=training)

        masks = keras.backend.in_train_phase(masks,
                                             lambda:detections(2, metadata, deltas, proposals, scores, masks),
                                             training=training)

        scores = keras.backend.in_train_phase(scores,
                                              lambda: detections(1, metadata, deltas, proposals, scores, masks),
                                              training=training)

        return [bounding_boxes, scores, masks]

    def compute_output_shape(self, input_shape):
        return [(1, input_shape[0][0], input_shape[1][2]),
                (1, input_shape[0][0], input_shape[2][2]),
                (1, input_shape[0][0], input_shape[2][2], input_shape[4][2], input_shape[4][3], input_shape[4][4])]

    def compute_mask(self, inputs, mask=None):
        return 3 * [None]

    @staticmethod
    def pad(x, padding):
        detections = keras.backend.shape(x)[1]

        difference = padding - detections

        difference = keras.backend.max([0, difference])

        paddings = ((0, 0), (0, difference), (0, 0))

        return tensorflow.pad(x, paddings, mode="constant")

    @staticmethod
    def padmasks(x, padding):
        detections = keras.backend.shape(x)[1]

        difference = padding - detections

        difference = keras.backend.max([0, difference])

        paddings = ((0, 0), (0, difference), (0, 0), (0, 0), (0, 0))

        return tensorflow.pad(x, paddings, mode="constant")


    def get_config(self):
        configuration = {
            "padding": self.padding
        }

        return {**super(ObjectDetection, self).get_config(), **configuration}
