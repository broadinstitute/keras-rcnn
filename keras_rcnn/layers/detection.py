import keras.backend
import keras.engine.topology

import keras_rcnn.backend


class Detection(keras.engine.topology.Layer):
    """
    Get final detections + labels by unscaling back to image space, applying
    regression deltas


    """

    def __init__(self, **kwargs):

        super(Detection, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Detection, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        # Inputs
        rois: output of proposal target (1, N, 4)
        pred_deltas: predicted deltas (1, N, 4*classes)
        pred_scores: score distributions (1, N, classes)
        metadata: image information (1, 3)

        # Returns
        pred_boxes: final predicted boxes of the predicted class (1, N, 4)
        pred_scores: score distribution over all classes (1, N, classes), note
        the box only corresponds to the most
            probable class, not the other classes
        """
        rois, pred_deltas, pred_scores, metadata = x[0], x[1], x[2], x[3]

        rois = keras.backend.reshape(rois, (-1, 4))
        pred_deltas = keras.backend.reshape(pred_deltas, (-1, keras.backend.shape(pred_deltas)[2]))

        # unscale back to raw image space
        boxes = rois / metadata[0][2]

        # Apply bounding-box regression deltas
        pred_boxes = keras_rcnn.backend.bbox_transform_inv(boxes, pred_deltas)

        pred_boxes = keras_rcnn.backend.clip(pred_boxes, metadata[0][:2])

        return [
            keras.backend.expand_dims(pred_boxes, 0),
            pred_scores
        ]

    def compute_output_shape(self, input_shape):
        return [(1, None, 4), (1, None, input_shape[2][2])]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]
