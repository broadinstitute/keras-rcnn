import keras.engine.topology
import tensorflow
import numpy
import keras_rcnn.backend


class ROI(keras.engine.topology.Layer):
    """ROI pooling layer proposed in Mask R-CNN (Kaiming He et. al.).

    # Parameters
    size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    regions: Integer, number of regions of interest.
    stride: Integer, pooling stride.

    # Returns
    4D Tensor (number of regions, slice_height, slice_width, channels)
    """

    def __init__(self, size, regions, stride=1, **kwargs):
        self.channels = None

        self.size = size

        self.regions = regions

        self.stride = stride

        super(ROI, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]

        super(ROI, self).build(input_shape)

    def call(self, x, **kwargs):
        image, regions = x[0], x[1]

        # convert regions from (x, y, w, h) to (x1, y1, x2, y2)
        regions = keras.backend.cast(regions, keras.backend.floatx())
        regions = regions / self.stride
        x1 = regions[:, 0]
        y1 = regions[:, 1]
        x2 = regions[:, 0] + regions[:, 2]
        y2 = regions[:, 1] + regions[:, 3]

        # normalize the boxes
        shape = keras.backend.int_shape(image)
        h = keras.backend.cast(shape[1], keras.backend.floatx())
        w = keras.backend.cast(shape[2], keras.backend.floatx())
        x1 /= w
        y1 /= h
        x2 /= w
        y2 /= h
        x1 = keras.backend.expand_dims(x1, axis=-1)
        y1 = keras.backend.expand_dims(y1, axis=-1)
        x2 = keras.backend.expand_dims(x2, axis=-1)
        y2 = keras.backend.expand_dims(y2, axis=-1)
        boxes = keras.backend.concatenate([y1, x1, y2, x2], axis=1)
        slices = keras_rcnn.backend.crop_and_resize(image, boxes, self.size)
        return slices

    def compute_output_shape(self, input_shape):
        return None, self.regions, self.size, self.size, self.channels
