# -*- coding: utf-8 -*-

import keras.preprocessing.image
import numpy
import skimage.color
import skimage.exposure
import skimage.io
import skimage.transform


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(
            self,
            dictionary,
            categories,
            target_size,
            generator,
            batch_size=1,
            color_mode="rgb",
            data_format=None,
            mask_size=(28, 28),
            seed=None,
            shuffle=False
    ):
        if color_mode not in {"grayscale", "rgb"}:
            raise ValueError

        self.batch_size = batch_size

        self.categories = categories

        if color_mode == "rgb":
            self.channels = 3
        else:
            self.channels = 1

        self.color_mode = color_mode

        if data_format is None:
            data_format = keras.backend.image_data_format()

        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError

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

        self.mask_size = mask_size

        self.maximum = numpy.max(target_size)

        self.minimum = numpy.min(target_size)

        self.n_categories = len(self.categories) + 1

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

    def find_scale(self, image):
        r, c, _ = image.shape

        scale = self.minimum / numpy.minimum(r, c)

        if numpy.maximum(r, c) * scale > self.maximum:
            scale = self.maximum / numpy.maximum(r, c)

        return scale

    def _get_batches_of_transformed_samples(self, selection):
        self.mask_size = (28, 28)

        target_bounding_boxes = numpy.zeros(
            (self.batch_size, 0, 4)
        )

        target_categories = numpy.zeros(
            (self.batch_size, 0, self.n_categories)
        )

        target_images = numpy.zeros(
            (self.batch_size, *self.target_size, self.channels)
        )

        target_masks = numpy.zeros(
            (self.batch_size, 0, *self.mask_size)
        )

        target_metadata = numpy.zeros(
            (self.batch_size, 3)
        )

        for batch_index, image_index in enumerate(selection):
            horizontal_flip = False

            if self.generator.horizontal_flip:
                if numpy.random.random() < 0.5:
                    horizontal_flip = True

            vertical_flip = False

            if self.generator.vertical_flip:
                if numpy.random.random() < 0.5:
                    vertical_flip = True

            pathname = self.dictionary[image_index]["image"]["pathname"]

            target_image = numpy.zeros((*self.target_size, self.channels))

            image = skimage.io.imread(pathname)

            scale = self.find_scale(image)

            image = skimage.transform.rescale(image, scale)

            image = self.generator.standardize(image)

            if horizontal_flip:
                image = numpy.fliplr(image)

            if vertical_flip:
                image = numpy.flipud(image)

            image_r = image.shape[0]
            image_c = image.shape[1]

            target_image[:image_r, :image_c] = image

            target_images[batch_index] = target_image

            target_metadata[batch_index] = [*self.target_size, 1.0]

            bounding_boxes = self.dictionary[image_index]["objects"]

            n_objects = len(bounding_boxes)

            target_bounding_boxes = numpy.resize(
                target_bounding_boxes, (self.batch_size, n_objects, 4)
            )

            target_masks = numpy.resize(
                target_masks, (self.batch_size, n_objects, *self.mask_size)
            )

            target_categories = numpy.resize(
                target_categories,
                (self.batch_size, n_objects, self.n_categories)
            )

            for bounding_box_index, bounding_box in enumerate(bounding_boxes):
                if bounding_box["category"] not in self.categories:
                    continue

                minimum_r = bounding_box["bounding_box"]["minimum"]["r"]
                minimum_c = bounding_box["bounding_box"]["minimum"]["c"]

                maximum_r = bounding_box["bounding_box"]["maximum"]["r"]
                maximum_c = bounding_box["bounding_box"]["maximum"]["c"]

                minimum_r *= scale
                minimum_c *= scale

                maximum_r *= scale
                maximum_c *= scale

                minimum_r = int(minimum_r)
                minimum_c = int(minimum_c)

                maximum_r = int(maximum_r)
                maximum_c = int(maximum_c)

                target_bounding_box = [
                    minimum_r,
                    minimum_c,
                    maximum_r,
                    maximum_c
                ]

                if horizontal_flip:
                    target_bounding_box = [
                        target_bounding_box[0],
                        image.shape[1] - target_bounding_box[3],
                        target_bounding_box[2],
                        image.shape[1] - target_bounding_box[1]
                    ]

                if vertical_flip:
                    target_bounding_box = [
                        image.shape[0] - target_bounding_box[2],
                        target_bounding_box[1],
                        image.shape[0] - target_bounding_box[0],
                        target_bounding_box[3]
                    ]

                target_bounding_boxes[
                    batch_index,
                    bounding_box_index
                ] = target_bounding_box

                if "mask" in bounding_box:
                    target_mask = skimage.io.imread(bounding_box["mask"]["pathname"])

                    target_mask = target_mask[minimum_r:maximum_r, minimum_c:maximum_c]

                    target_mask = skimage.transform.resize(target_mask, self.mask_size, order=0)

                    if horizontal_flip:
                        target_mask = numpy.fliplr(target_mask)

                    if vertical_flip:
                        target_mask = numpy.flipud(target_mask)

                    target_masks[
                        batch_index,
                        bounding_box_index
                    ] = target_mask

                target_category = numpy.zeros((self.n_categories))

                target_category[self.categories[bounding_box["category"]]] = 1

                target_categories[
                    batch_index,
                    bounding_box_index
                ] = target_category

        x = [
            target_bounding_boxes,
            target_categories,
            target_images,
            target_masks,
            target_metadata
        ]

        return x, None


class ObjectDetectionGenerator:
    def __init__(
            self,
            data_format=None,
            horizontal_flip=False,
            preprocessing_function=None,
            rescale=False,
            rotation_range=0.0,
            samplewise_center=False,
            vertical_flip=False
    ):
        self.data_format = data_format

        self.horizontal_flip = horizontal_flip

        self.preprocessing_function = preprocessing_function

        self.rescale = rescale

        self.rotation_range = rotation_range

        self.samplewise_center = samplewise_center

        self.vertical_flip = vertical_flip

    def flow_from_dictionary(
            self,
            dictionary,
            categories,
            target_size,
            batch_size=1,
            color_mode="rgb",
            data_format=None,
            mask_size=(28, 28),
            shuffle=True,
            seed=None
    ):
        return DictionaryIterator(
            dictionary,
            categories,
            target_size,
            self,
            batch_size,
            color_mode,
            data_format,
            mask_size,
            seed,
            shuffle
        )

    def standardize(self, image):
        if self.preprocessing_function:
            image = self.preprocessing_function(image)

        if self.rescale:
            image *= self.rescale

        if self.samplewise_center:
            image -= numpy.mean(image, keepdims=True)

        image = skimage.exposure.rescale_intensity(image, out_range=(0.0, 1.0))

        return image
