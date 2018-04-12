import os.path
import io
import keras.callbacks
import matplotlib.pyplot
import numpy
import tensorflow

import keras_rcnn.utils


# TODO: use skimage.draw to circumvent matplotlib
def _generate_image(image, bounding_boxes):
    figure = matplotlib.pyplot.figure()

    axis = figure.gca()

    axis.set_axis_off()

    bbox_inches = axis.get_window_extent().transformed(matplotlib.pyplot.gcf().dpi_scale_trans.inverted())

    keras_rcnn.utils.show_bounding_boxes(image, bounding_boxes)

    buffer = io.BytesIO()

    matplotlib.pyplot.savefig(buffer, bbox_inches=bbox_inches)

    buffer.seek(0)

    return buffer


class TensorBoard(keras.callbacks.TensorBoard):
    """

    """

    def __init__(self, generator):
        self.generator = generator

        super(TensorBoard, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        score = 0.0

        summary = tensorflow.Summary()

        summary_value = summary.value.add()

        summary_value.simple_value = score

        summary_value.tag = "mean average precision (mAP)"

        self.writer.add_summary(summary, epoch)

        summary = self._summarize_image()

        summary = keras.backend.eval(summary)

        self.writer.add_summary(summary)

    def _summarize_image(self):
        shape = (self.generator.n, *self.generator.target_size, self.generator.channels)

        images = numpy.zeros((self.generator.n, 224, 341, 4))

        for generator_index in range(self.generator.n):
            x, _ = self.generator.next()

            target_bounding_boxes, _, target_images, _, _ = x

            buffer = _generate_image(target_images[0], target_bounding_boxes[0])

            image = tensorflow.image.decode_png(buffer.getvalue(), channels=4)

            image = tensorflow.expand_dims(image, 0)

            image = keras.backend.eval(image)

            images[generator_index] = image

        return tensorflow.summary.image("training", images)
