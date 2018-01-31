# -*- coding: utf-8 -*-

import keras.preprocessing.image
import numpy
import skimage.color
import skimage.io


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(
            self,
            dictionary,
            classes,
            target_size,
            generator,
            batch_size=1,
            color_mode="rgb",
            data_format=None,
            shuffle=False,
            seed=None
    ):
        self.batch_size = batch_size

        self.classes = classes

        if color_mode not in {"grayscale", "rgb"}:
            raise ValueError("Invalid color mode:", color_mode, "; expected 'grayscale' or 'rgb'.")

        self.color_mode = color_mode

        if data_format is None:
            data_format = keras.backend.image_data_format()

        self.data_format = data_format

        self.dictionary = dictionary

        self.generator = generator

        if self.color_mode == "grayscale":
            if self.data_format == "channels_last":
                self.image_shape = target_size + (1,)
            else:
                self.image_shape = (1,) + target_size
        else:
            if self.data_format == "channels_last":
                self.image_shape = target_size + (3,)
            else:
                self.image_shape = (3,) + target_size

        self.target_size = target_size

        super(DictionaryIterator, self).__init__(len(self.dictionary), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            selection = next(self.index_generator)

        return self._get_batches_of_transformed_samples(selection)

    def _get_batches_of_transformed_samples(self, selection):
        # Labels has num_classes + 1 elements, since 0 is reserved for
        # background.
        num_classes = len(self.classes) + 1

        target_bounding_boxes = numpy.zeros((self.batch_size, 0, 4))

        if self.color_mode == "rgb":
            channels = 3
        else:
            channels = 1

        target_images = numpy.zeros((self.batch_size, *self.target_size, channels))

        target_labels = numpy.zeros((self.batch_size, 0, num_classes))

        target_metadata = numpy.zeros((self.batch_size, 3))

        for batch_index, image_index in enumerate(selection):
            count = 0

            while count == 0:
                pathname = self.dictionary[image_index]["filename"]

                image = skimage.io.imread(pathname)

                image = self.generator.standardize(image)

                target_images[batch_index] = image

                target_metadata[batch_index] = [*self.target_size, 1.0]

                bounding_boxes = self.dictionary[image_index]["boxes"]

                n = len(bounding_boxes)

                target_bounding_boxes = numpy.resize(target_bounding_boxes, (self.batch_size, n, 4))

                target_labels = numpy.resize(target_labels, (self.batch_size, n, num_classes))

                for bounding_box_index, bounding_box in enumerate(bounding_boxes):
                    if bounding_box["class"] not in self.classes:
                        continue

                    minimum_c = bounding_box["x1"]
                    maximum_c = bounding_box["x2"]
                    minimum_r = bounding_box["y1"]
                    maximum_r = bounding_box["y2"]

                    if maximum_c == image.shape[1]:
                        maximum_c -= 1

                    if maximum_r == image.shape[0]:
                        maximum_r -= 1

                    if minimum_c >= 0 and maximum_c < image.shape[1] and minimum_r >= 0 and maximum_r < image.shape[0]:
                        count += 1

                        target_bounding_box = [minimum_c, minimum_r, maximum_c, maximum_r]

                        target_bounding_boxes[batch_index, bounding_box_index] = target_bounding_box

                        # Store the labels in one-hot form.
                        target_label = [0] * (num_classes)

                        target_label[self.classes[bounding_box["class"]]] = 1

                        target_labels[batch_index, bounding_box_index] = target_label

        return [target_bounding_boxes, target_images, target_labels, target_metadata], None


class ObjectDetectionGenerator:
    def __init__(
            self,
            data_format=None,
            preprocessing_function=None,
            rescale=False,
            samplewise_center=False
    ):
        if data_format is None:
            data_format = keras.backend.image_data_format()

        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError

        self.data_format = data_format

        self.preprocessing_function = preprocessing_function

        self.rescale = rescale

        self.samplewise_center = samplewise_center

    def flow_from_dictionary(
            self,
            dictionary,
            classes,
            target_size,
            batch_size=1,
            color_mode="rgb",
            shuffle=True,
            seed=None
    ):
        return DictionaryIterator(
            dictionary,
            classes,
            target_size,
            self,
            batch_size,
            color_mode,
            shuffle,
            seed
        )

    def standardize(self, image):
        if self.preprocessing_function:
            image = self.preprocessing_function(image)

        if self.rescale:
            image *= self.rescale

        if self.samplewise_center:
            image -= numpy.mean(image, keepdims=True)

        return image
