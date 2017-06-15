import keras.engine.topology

import keras_rcnn.backend


class ROI(keras.engine.topology.Layer):
    """
    ROI pooling layer proposed in Mask R-CNN (Kaiming He et. al.).

    :param size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    :param stride: Integer, pooling stride.
    :return: slices: 5D Tensor (number of regions, slice_height,
    slice_width, channels)
    """
    def __init__(self, size, stride=1, **kwargs):
        self.channels = None

        self.size = size

        self.stride = stride

        super(ROI, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]

        super(ROI, self).build(input_shape)

    def call(self, x, **kwargs):
        image, boxes = x[0], x[1]

        # convert regions from (x, y, w, h) to (x1, y1, x2, y2)
        boxes = keras.backend.cast(boxes, keras.backend.floatx())

        boxes = boxes / self.stride

        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 0] + boxes[..., 2]
        y2 = boxes[..., 1] + boxes[..., 3]

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

        boxes = keras.backend.concatenate([y1, x1, y2, x2], axis=-1)

        slices = keras_rcnn.backend.crop_and_resize(image, boxes, self.size)

        return keras.backend.expand_dims(slices, axis=0)

    def compute_output_shape(self, input_shape):
        return 1, None, self.size[0], self.size[1], self.channels
