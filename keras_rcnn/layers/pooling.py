import keras.engine.topology
import tensorflow


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
