import keras.engine.topology
import tensorflow

import keras_rcnn.backend
from .pooling import ROI


class Proposal(keras.engine.topology.Layer):
    def __init__(self, proposals, **kwargs):
        self.output_dim = (None, None, 4)

        self.proposals = proposals

        super(Proposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Proposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return propose(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return self.output_dim


def propose(boxes, scores):
    shape = keras.backend.int_shape(boxes)[1:3]

    shifted = keras_rcnn.backend.shift(shape, 16)

    proposals = keras.backend.reshape(boxes, (-1, 4))

    proposals = keras_rcnn.backend.bbox_transform_inv(shifted, proposals)

    proposals = keras_rcnn.backend.clip(proposals, shape)

    indicies = keras_rcnn.backend.filter_boxes(proposals, 1)

    proposals = keras.backend.gather(proposals, indicies)

    scores = scores[:, :, :, :9]
    scores = keras.backend.reshape(scores, (-1, 1))
    scores = keras.backend.gather(scores, indicies)
    scores = keras.backend.flatten(scores)

    proposals = keras.backend.cast(proposals, tensorflow.float32)
    scores = keras.backend.cast(scores, tensorflow.float32)

    indicies = keras_rcnn.backend.non_maximum_suppression(proposals, scores, 100, 0.7)

    proposals = keras.backend.gather(proposals, indicies)

    return keras.backend.expand_dims(proposals, 0)
