# -*- coding: utf-8 -*-

import keras.preprocessing.image
import numpy
import skimage.color
import skimage.exposure
import skimage.io
import skimage.transform


def find_scale(image, minimum=1024, maximum=1024):
    (rows, cols, _) = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = minimum / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > maximum:
        scale = maximum / largest_side

    return scale


def flip_axis(image, axis):
    image = numpy.asarray(image).swapaxes(axis, 0)
    image = image[::-1, ...]
    image = image.swapaxes(0, axis)

    return image


def flip_bounding_boxes_horizontally(bounding_boxes):
    pass


def flip_bounding_boxes_vertically(bounding_boxes):
    pass


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
            minimum=256,
            maximum=512,
            seed=None,
            shuffle=False
    ):
        if color_mode not in {"grayscale", "rgb"}:
            raise ValueError

        self.batch_size = batch_size

        self.classes = classes

        if color_mode == "rgb":
            self.channels = 3
        else:
            self.channels = 1

        self.color_mode = color_mode

        if data_format is None:
            data_format = keras.backend.image_data_format()

        self.data_format = data_format

        self.dictionary = dictionary

        self.generator = generator

        if self.color_mode == "grayscale":
            if self.data_format == "channels_first":
                self.image_shape = (*target_size, 1)
            else:
                self.image_shape = (1, *target_size)
        else:
            if self.data_format == "channels_last":
                self.image_shape = (*target_size, 3)
            else:
                self.image_shape = (3, *target_size)

        if self.data_format == "channels_first":
            self.mask_shape = (*target_size, 1)
        else:
            self.mask_shape = (1, *target_size)

        self.n_classes = len(self.classes) + 1

        self.n_samples = len(self.dictionary)

        self.target_size = target_size

        super(DictionaryIterator, self).__init__(
            self.n_samples,
            batch_size,
            shuffle,
            seed
        )

    def next(self):
        with self.lock:
            selection = next(self.index_generator)

        return self._get_batches_of_transformed_samples(selection)

    def _get_batches_of_transformed_samples(self, selection):
        target_bounding_boxes = numpy.zeros(
            (self.batch_size, 0, 4)
        )

        target_images = numpy.zeros(
            (self.batch_size, *self.target_size, self.channels)
        )

        target_masks = numpy.zeros(
            (self.batch_size, *self.target_size, 1)
        )

        target_metadata = numpy.zeros(
            (self.batch_size, 3)
        )

        target_scores = numpy.zeros(
            (self.batch_size, 0, self.n_classes)
        )

        for batch_index, image_index in enumerate(selection):
            pathname = self.dictionary[image_index]["image"]["pathname"]

            target_image = numpy.zeros((*self.target_size, self.channels))

            image = skimage.io.imread(pathname)

            scale = find_scale(image)

            image = skimage.transform.rescale(image, scale)

            image = self.generator.standardize(image)

            image = self.generator.random_transform(image)

            c = image.shape[0]
            r = image.shape[1]

            target_image[:c, :r] = image

            target_images[batch_index] = target_image

            target_metadata[batch_index] = [*self.target_size, 1.0]

            bounding_boxes = self.dictionary[image_index]["objects"]

            n_objects = len(bounding_boxes)

            target_bounding_boxes = numpy.resize(
                target_bounding_boxes, (self.batch_size, n_objects, 4)
            )

            target_masks = numpy.resize(
                target_masks, (self.batch_size, n_objects, *self.target_size)
            )

            target_scores = numpy.resize(
                target_scores, (self.batch_size, n_objects, self.n_classes)
            )

            for bounding_box_index, bounding_box in enumerate(bounding_boxes):
                if bounding_box["class"] not in self.classes:
                    continue

                c = image.shape[0]
                r = image.shape[1]

                minimum_c = bounding_box["bounding_box"]["minimum"]["c"]
                minimum_r = bounding_box["bounding_box"]["minimum"]["r"]

                maximum_c = bounding_box["bounding_box"]["maximum"]["c"]
                maximum_r = bounding_box["bounding_box"]["maximum"]["r"]

                minimum_c *= scale
                minimum_r *= scale

                maximum_c *= scale
                maximum_r *= scale

                if minimum_c >= 0 and minimum_r >= 0 and maximum_c < r and maximum_r < c:
                    target_bounding_box = [
                        minimum_c,
                        minimum_r,
                        maximum_c,
                        maximum_r
                    ]

                    target_bounding_boxes[
                        batch_index,
                        bounding_box_index
                    ] = target_bounding_box

                    target_mask = numpy.zeros(self.target_size)

                    mask = skimage.io.imread(bounding_box["mask"]["pathname"])

                    mask = skimage.transform.rescale(mask, scale, 0)

                    mask_c = image.shape[0]
                    mask_r = image.shape[1]

                    target_mask[:mask_c, :mask_r] = mask

                    target_masks[
                        batch_index,
                        bounding_box_index
                    ] = target_mask

                    target_score = numpy.zeros((self.n_classes))

                    target_score[self.classes[bounding_box["class"]]] = 1

                    target_scores[
                        batch_index,
                        bounding_box_index
                    ] = target_score

        x = [
            target_bounding_boxes,
            target_images,
            target_masks,
            target_metadata,
            target_scores
        ]

        return x, None


class ObjectDetectionGenerator:
    def __init__(
            self,
            data_format=None,
            horizontal_flip=False,
            preprocessing_function=None,
            rescale=False,
            samplewise_center=False,
            vertical_flip=False
    ):
        if data_format is None:
            data_format = keras.backend.image_data_format()

        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError

        self.data_format = data_format

        self.horizontal_flip = horizontal_flip

        self.preprocessing_function = preprocessing_function

        self.rescale = rescale

        self.samplewise_center = samplewise_center

        self.vertical_flip = vertical_flip

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

    def random_transform(self, image, seed=None):
        if seed is not None:
            numpy.random.seed(seed)

        horizontal_flip = False

        if self.horizontal_flip:
            if numpy.random.random() < 0.5:
                horizontal_flip = True

        vertical_flip = False

        if self.vertical_flip:
            if numpy.random.random() < 0.5:
                vertical_flip = True

        if horizontal_flip:
            image = flip_axis(image, 1)

        if vertical_flip:
            image = flip_axis(image, 0)

        return image

    def standardize(self, image):
        if self.preprocessing_function:
            image = self.preprocessing_function(image)

        if self.rescale:
            image *= self.rescale

        if self.samplewise_center:
            image -= numpy.mean(image, keepdims=True)

        image = skimage.exposure.rescale_intensity(image, out_range=(0.0, 1.0))

        return image
