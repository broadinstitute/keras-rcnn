import keras.backend
import keras.preprocessing.image
import numpy
import skimage.transform
import skimage.io
import time

def scale_size(size, min_size=224, max_size=224):
    """
    Rescales a given image size such that the larger axis is
    no larger than max_size and the smallest axis is as close
    as possible to min_size.
    """
    assert(len(size) == 2)

    scale = min_size / numpy.min(size)

    # Prevent the biggest axis from being larger than max_size.
    if numpy.round(scale * numpy.max(size)) > max_size:
        scale = max_size / numpy.max(size)

    rows, cols = size
    rows *= scale
    cols *= scale

    return (int(rows), int(cols)), scale


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(self, dictionary, classes, generator, batch_size=1, shuffle=False, seed=None):
        self.dictionary = dictionary
        self.classes    = classes
        self.generator  = generator

        assert(len(self.dictionary) != 0)

        # Compute and store the target image shape.
        cols, rows, channels          = dictionary[0]["shape"]
        self.image_shape              = (rows, cols, channels)

        self.target_shape, self.scale = scale_size(self.image_shape[0:2])
        self.target_shape             = self.target_shape + (self.image_shape[2],)

        # Metadata needs to be computed only once.
        rows, cols, channels          = self.target_shape
        self.metadata                 = numpy.array([[rows, cols, self.scale]])

        super().__init__(len(self.dictionary), batch_size, shuffle, seed)

    def next(self):
        # Lock indexing to prevent race conditions.
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        # Labels has num_classes + 1 elements, since 0 is reserved for background.
        num_classes = len(self.classes)
        images      = numpy.zeros((batch_size,) + self.target_shape, dtype=keras.backend.floatx())
        boxes       = numpy.zeros((batch_size, 0, 4),                dtype=keras.backend.floatx())
        labels      = numpy.zeros((batch_size, 0, num_classes + 1),  dtype=numpy.uint8)

        for batch_index, image_index in enumerate(selection):
            path  = self.dictionary[image_index]["filename"]
            image = skimage.io.imread(path)

            # Assert that the loaded image has the predefined image shape.
            if image.shape != self.image_shape:
                raise Exception("All input images need to be of the same shape.")

            # Copy image to batch blob.
            images[batch_index] = skimage.transform.rescale(image, scale=self.scale, mode="reflect")

            # Set ground truth boxes.
            for i, b in enumerate(self.dictionary[image_index]["boxes"]):
                if b["class"] not in self.classes:
                    raise Exception("Class {} not found in '{}'.".format(b["class"], self.classes))

                box   = [b["y1"], b["x1"], b["y2"], b["x2"]]
                boxes = numpy.append(boxes, [[box]], axis=1)

                # Store the labels in one-hot form.
                label = [0] * (num_classes + 1)
                label[self.classes[b["class"]]] = 1
                labels = numpy.append(labels, [[label]], axis = 1)

            # Scale the ground truth boxes to the selected image scale.
            boxes[batch_index, :, :4] *= self.scale

        return [images, boxes, self.metadata, labels], None

class ObjectDetectionGenerator:
    def flow(self, dictionary, classes):
        return DictionaryIterator(dictionary, classes, self)
