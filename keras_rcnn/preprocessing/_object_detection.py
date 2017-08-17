import keras.backend
import keras.preprocessing.image
import numpy
import skimage.transform
import skimage.io
import time

def scale_shape(shape, min_size=224, max_size=224):
    """
    Rescales a given shape such that the larger axis is no
    larger than max_size and the smallest axis is as close
    as possible to min_size.
    """
    min_shape = numpy.min(shape[0:2])
    max_shape = numpy.max(shape[0:2])
    scale     = min_size / min_shape

    # Prevent the biggest axis from being larger than max_size
    if numpy.round(scale * max_shape) > max_size:
        scale = max_size / max_shape

    return (int(shape[0] * scale), int(shape[1] * scale), shape[2]), scale


def scale_image(image, min_size=224, max_size=224):
    """
    Rescales an image according to the heuristics from 'scale_shape'.
    """
    target_shape, scale = scale_shape(image.shape, min_size, max_size)

    return skimage.transform.rescale(image, scale=scale, mode="reflect"), scale


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(self, data, classes, image_data_generator, image_shape, batch_size=1,
                 shuffle=True, seed=numpy.uint32(time.time() * 1000)):
        self.data                 = data
        self.classes              = classes
        self.image_data_generator = image_data_generator
        self.image_shape          = image_shape
        self.target_shape, _      = scale_shape(image_shape)

        super().__init__(len(self.data), batch_size, shuffle, seed)

    def next(self):
        # Lock indexing to prevent race conditions
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        # Transformation of images is not under thread lock so it can be done in parallel
        image_batch    = numpy.zeros((batch_size,) + self.target_shape, dtype=keras.backend.floatx())
        gt_boxes_batch = numpy.zeros((batch_size, 0, 5),                dtype=keras.backend.floatx())
        metadata       = numpy.zeros((batch_size, 3),                   dtype=keras.backend.floatx())

        for batch_index, image_index in enumerate(selection):
            path         = self.data[image_index]["filename"]
            image        = skimage.io.imread(path, as_grey=(self.target_shape[2] == 1))
            image, scale = scale_image(image)
            image        = self.image_data_generator.random_transform(image)
            image        = self.image_data_generator.standardize(image)

            # Copy image to batch blob
            image_batch[batch_index] = image

            # Set ground truth boxes
            boxes = self.data[image_index]["boxes"]
            for i, b in enumerate(boxes):
                if b["class"] not in self.classes:
                    raise Exception("Class {} not found in '{}'.".format(b["class"], self.classes))

                gt_data = [b["y1"], b["x1"], b["y2"], b["x2"], self.classes[b["class"]]]
                gt_boxes_batch = numpy.append(gt_boxes_batch, [[gt_data]], axis=1)

            # Scale the ground truth boxes to the selected image scale
            gt_boxes_batch[batch_index, :, :4] *= scale

            # Create metadata
            metadata[batch_index, :] = [image.shape[0], image.shape[1], scale]

        return [image_batch, gt_boxes_batch, metadata], None

class ObjectDetectionGenerator:
    def flow(self, data, classes, image_shape):
        return DictionaryIterator(data, classes, keras.preprocessing.image.ImageDataGenerator(), image_shape)
