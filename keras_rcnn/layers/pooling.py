import keras.engine.topology
import tensorflow
import numpy
import keras_rcnn.backend


class ROI(keras.engine.topology.Layer):
    def __init__(self, size, regions, **kwargs):
        self.channels = None

        self.size = size

        self.regions = regions

        super(ROI, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]

        super(ROI, self).build(input_shape)

    def call(self, x, **kwargs):
        image = x[0]

        regions = x[1]

        outputs = []

        for index in range(self.regions):
            x = regions[0, index, 0]
            y = regions[0, index, 1]
            w = regions[0, index, 2]
            h = regions[0, index, 3]

            x = keras.backend.cast(x, "int32")
            y = keras.backend.cast(y, "int32")
            w = keras.backend.cast(w, "int32")
            h = keras.backend.cast(h, "int32")

            image = image[:, y:y + h, x:x + w, :]

            shape = (self.size, self.size)

            resized = tensorflow.image.resize_images(image, shape)

            outputs.append(resized)

        y = keras.backend.concatenate(outputs, axis=0)

        shape = (1, self.regions, self.size, self.size, self.channels)

        y = keras.backend.reshape(y, shape)

        pattern = (0, 1, 2, 3, 4)

        return keras.backend.permute_dimensions(y, pattern)

    def compute_output_shape(self, input_shape):
        return None, self.regions, self.size, self.size, self.channels


class ROIAlign(ROI):
    """ROIAlign pooling layer proposed in Mask R-CNN (Kaiming He et. al.).

    # Parameters
    size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    regions: Integer, number of regions of interest.
    stride: Integer, pooling stride.

    # Returns
    4D Tensor (number of regions, slice_height, slice_width, channels)
    """

    def __init__(self, size, regions, stride=1, **kwargs):
        self.stride = stride
        super(ROIAlign, self).__init__(size=size,
                                       regions=regions,
                                       **kwargs)

    def call(self, x, **kwargs):
        image, regions = x[0], x[1]

        # convert regions from (x, y, h, w) to (x1, y1, x2, y2)
        regions = keras.backend.cast(regions, keras.backend.floatx())
        regions = regions / (self.stride + 0.0)
        x1 = regions[:, 0]
        y1 = regions[:, 1]
        x2 = regions[:, 0] + regions[:, 3]
        y2 = regions[:, 1] + regions[:, 2]

        # normalize the boxes
        shape = image._keras_shape
        h = keras.backend.cast(shape[1], keras.backend.floatx())
        w = keras.backend.cast(shape[2], keras.backend.floatx())
        x1 /= w
        y1 /= h
        x2 /= w
        y2 /= h
        boxes = keras.backend.concatenate([y1[:, numpy.newaxis],
                                           x1[:, numpy.newaxis],
                                           y2[:, numpy.newaxis],
                                           x2[:, numpy.newaxis]], axis=1)
        slices = keras_rcnn.backend.crop_and_resize(image, boxes, self.size)
        return slices
