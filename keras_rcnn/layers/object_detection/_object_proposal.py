import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend


class ObjectProposal(keras.engine.topology.Layer):
    def __init__(self, maximum_proposals=300, **kwargs):
        self.output_dim = (None, maximum_proposals, 4)

        self.maximum_proposals = maximum_proposals

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.propose(inputs[0], inputs[1], self.maximum_proposals)

    def compute_output_shape(self, input_shape):
        return self.output_dim

    @staticmethod
    def propose(boxes, scores, maximum):
        shape = keras.backend.shape(boxes)[1:3]

        shifted = keras_rcnn.backend.shift(shape, 16)

        proposals = keras.backend.reshape(boxes, (-1, 4))

        proposals = keras_rcnn.backend.bbox_transform_inv(shifted, proposals)

        proposals = keras_rcnn.backend.clip(proposals, shape)

        indices = keras_rcnn.backend.filter_boxes(proposals, 1)

        proposals = keras.backend.gather(proposals, indices)
        scores = scores[:, :, :, :9]
        scores = keras.backend.reshape(scores, (-1, 1))
        scores = keras.backend.gather(scores, indices)
        scores = keras.backend.flatten(scores)

        proposals = keras.backend.cast(proposals, keras.backend.floatx())
        scores = keras.backend.cast(scores, keras.backend.floatx())

        indices = keras_rcnn.backend.non_maximum_suppression(proposals, scores, maximum, 0.7)

        proposals = keras.backend.gather(proposals, indices)

        return keras.backend.expand_dims(proposals, 0)
