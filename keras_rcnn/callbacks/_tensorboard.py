import os.path

import keras.callbacks
import matplotlib.pyplot
import numpy
import tensorflow

import keras_rcnn.utils


def _save_images(generator, pathname):
    generator.shuffle = False

    matplotlib.pyplot.ioff()

    for generator_index in range(generator.n):
        (target_bounding_boxes, _, target_images, _, _), _ = generator.next()

        batch_size = target_bounding_boxes.shape[0]

        for batch_index in range(batch_size):
            figure = matplotlib.pyplot.figure(
                figsize=(8, 8)
            )

            axis = figure.gca()

            axis.set_axis_off()

            keras_rcnn.utils.show_bounding_boxes(
                target_images[batch_index],
                target_bounding_boxes[batch_index]
            )

            filename = "{}-{}.png".format(generator_index, batch_index)

            window_extent = axis.get_window_extent()

            inverted = axis.dpi_scale_trans.inverted()

            bbox_inches = window_extent.transformed(inverted)

            matplotlib.pyplot.savefig(
                os.path.join(pathname, filename),
                bbox_inches=bbox_inches
            )

            matplotlib.pyplot.close(figure)


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
        shape = (
            self.generator.n,
            *self.generator.target_size,
            self.generator.channels
        )

        images = numpy.zeros(shape)

        for generator_index in range(self.generator.n):
            x, _ = self.generator.next()

            target_bounding_boxes, _, target_images, _, _ = x

            images[generator_index] = target_images

        images = keras.backend.variable(images)

        return tensorflow.summary.image("training", images)
