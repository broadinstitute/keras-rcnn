# -*- coding: utf-8 -*-

import keras.engine.topology

import keras_rcnn.backend


class RegionOfInterest(keras.engine.topology.Layer):
    """
    ROI pooling layer proposed in Mask R-CNN (Kaiming He et. al.).

    :param size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    :param stride: Integer, pooling stride.
    :return: slices: 5D Tensor (number of regions, slice_height,
    slice_width, channels)
    """
    def __init__(self, extent=(14, 14), stride=1, **kwargs):
        self.channels = None

        self.extent = extent

        self.stride = stride

        super(RegionOfInterest, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[1][3]

        super(RegionOfInterest, self).build(input_shape)

    def call(self, x, **kwargs):
        """

        :rtype: `(samples, proposals, width, height, channels)`
        """
        metadata, image, boxes = x[0], x[1], x[2]

        # convert regions from (x, y, w, h) to (x1, y1, x2, y2)
        boxes = keras.backend.cast(boxes, keras.backend.floatx())

        boxes = boxes / self.stride

        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        # normalize the boxes
        shape = metadata[0]

        h = keras.backend.cast(shape[0], keras.backend.floatx())
        w = keras.backend.cast(shape[1], keras.backend.floatx())

        x1 /= w
        y1 /= h
        x2 /= w
        y2 /= h

        x1 = keras.backend.expand_dims(x1, axis=2)
        y1 = keras.backend.expand_dims(y1, axis=2)
        x2 = keras.backend.expand_dims(x2, axis=2)
        y2 = keras.backend.expand_dims(y2, axis=2)

        boxes = keras.backend.concatenate([y1, x1, y2, x2], axis=2)
        boxes = keras.backend.reshape(boxes, (-1, 4))

        slices = keras_rcnn.backend.crop_and_resize(image, boxes, self.extent)

        return keras.backend.expand_dims(slices, axis=0)

    def compute_output_shape(self, input_shape):
        return 1, input_shape[2][1], self.extent[0], self.extent[1], self.channels

    def get_config(self):
        configuration = {
            "extent": self.extent,
            "stride": self.stride
        }

        return {**super(RegionOfInterest, self).get_config(), **configuration}
