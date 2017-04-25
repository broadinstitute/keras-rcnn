import keras.engine.topology
import tensorflow


class ROI(keras.engine.topology.Layer):
    def __init__(self, pool_size, regions_of_interest, **kwargs):
        self.channels = None

        self.pool_size = pool_size

        self.regions_of_interest = regions_of_interest

        super(ROI, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]

        super(ROI, self).build(input_shape)

    def call(self, x, **kwargs):
        image = x[0]

        regions_of_interest = x[1]

        outputs = []

        for index, _ in enumerate(self.regions_of_interest):
            x = regions_of_interest[0, index, 0]
            y = regions_of_interest[0, index, 1]
            w = regions_of_interest[0, index, 2]
            h = regions_of_interest[0, index, 3]

            x = keras.backend.cast(x, "int32")
            y = keras.backend.cast(y, "int32")
            w = keras.backend.cast(w, "int32")
            h = keras.backend.cast(h, "int32")

            image = image[:, y:y + h, x:x + w, :]

            shape = (self.pool_size, self.pool_size)

            resized = tensorflow.image.resize_images(image, shape)

            outputs.append(resized)

        y = keras.backend.concatenate(outputs, axis=0)

        shape = (
            1,
            self.regions_of_interest,
            self.pool_size,
            self.pool_size,
            self.channels
        )

        y = keras.backend.reshape(y, shape)

        pattern = (0, 1, 2, 3, 4)

        return keras.backend.permute_dimensions(y, pattern)

    def compute_output_shape(self, input_shape):
        return (
            None,
            self.regions_of_interest,
            self.pool_size,
            self.pool_size,
            self.channels
        )


class SpatialPyramid(keras.engine.topology.Layer):
    def __init__(self, **kwargs):
        super(SpatialPyramid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpatialPyramid, self).build(input_shape)

    def call(self, x, **kwargs):
        return

    def compute_output_shape(self, input_shape):
        return
