# -*- coding: utf-8 -*-

import keras.engine.topology

import keras_rcnn.backend


class RegionOfInterest(keras.layers.Layer):
    """
    ROI pooling layer proposed in Mask R-CNN (Kaiming He et. al.).

    :param size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    :param stride: Integer, pooling stride.
    :return: slices: 5D Tensor (number of regions, slice_height,
    slice_width, channels)
    """
    def __init__(self, extent=(7, 7), strides=1, **kwargs):
        self.channels = None

        self.extent = extent

        self.stride = strides

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

        x1 /= w - 1
        y1 /= h - 1
        x2 /= w - 1
        y2 /= h - 1

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
            "strides": self.stride
        }

        return {**super(RegionOfInterest, self).get_config(), **configuration}


def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return keras.backend.log(x) / keras.backend.log(2.0)


class RegionOfInterestAlignPyramid(keras.layers.Layer):
    def __init__(self, extent=(7, 7), strides=1, **kwargs):
        self.channels = None

        self.extent = extent

        self.stride = strides

        super(RegionOfInterestAlignPyramid, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[2][3]

        super(RegionOfInterestAlignPyramid, self).build(input_shape)

    def call(self, x, **kwargs):
        metadata, boxes, images = x[0], x[1], x[2:]

        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        h = y2 - y1
        w = x2 - x1

        image_rows = metadata[0, 0]
        image_cols = metadata[0, 1]

        image_area = keras.backend.cast(image_rows * image_cols, 'float32')
        roi_level = log2_graph(keras.backend.sqrt(h * w) / (keras.backend.sqrt(image_area)))
        roiInt = keras.backend.round(roi_level)
        roi_level = keras.backend.minimum(5.0, keras.backend.maximum(2.0, 4.0 + roiInt))
        roi_level = keras.backend.squeeze(roi_level, 0)

        pooled = []
        box_to_level = []
        for i in range(0, len(images)):
            level = i + 2

            ix = keras_rcnn.backend.where(keras.backend.equal(roi_level, level))

            level_boxes = keras.backend.tf.gather_nd(keras.backend.squeeze(boxes, axis=0), ix)

            level_boxes = keras.backend.expand_dims(level_boxes, axis=0)

            pool = keras_rcnn.layers.RegionOfInterest(
                extent=self.extent,
                strides=self.stride
            )([
                metadata,
                images[i],
                level_boxes
            ])

            pooled.append(pool)
            box_to_level.append(ix)

        pooled = keras.backend.concatenate(pooled, axis=1)
        box_to_level = keras.backend.concatenate(box_to_level, axis=0)

        box_range = keras.backend.expand_dims(keras.backend.tf.range(keras.backend.shape(box_to_level)[0]), 1)
        box_to_level_int = keras.backend.cast(keras.backend.round(box_to_level), 'int32')
        box_to_level = keras.backend.concatenate([box_to_level_int, box_range], axis=1)

        pooled = keras.backend.squeeze(pooled, axis=0)
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = keras.backend.tf.nn.top_k(sorting_tensor, k=keras.backend.shape(
            box_to_level)[0]).indices[::-1]

        ix = keras.backend.gather(box_to_level[:, 1], ix)

        pooled = keras.backend.gather(pooled, ix)

        shape = keras.backend.concatenate([keras.backend.shape(boxes)[:2], keras.backend.shape(pooled)[1:]], axis=0)
        pooled = keras.backend.reshape(pooled, shape)

        keras.backend.expand_dims(pooled, axis=0)

        return pooled

    def compute_output_shape(self, input_shape):
        return 1, input_shape[1][1], self.extent[0], self.extent[1], self.channels

    def get_config(self):
        configuration = {
            "extent": self.extent,
            "strides": self.stride
        }

        return {**super(RegionOfInterestAlignPyramid, self).get_config(), **configuration}
