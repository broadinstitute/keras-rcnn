# -*- coding: utf-8 -*-

import keras.preprocessing.image
import numpy
import skimage.color
import skimage.exposure
import skimage.io
import skimage.transform


class BoundingBoxException(Exception):
    pass


class MissingImageException(Exception):
    pass


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

    def _clear_border(self, bounding_boxes):
        indices = []

        for index, bounding_box in enumerate(bounding_boxes[0]):
            minimum_r, minimum_c, maximum_r, maximum_c = bounding_box

            if minimum_r > 0 and minimum_c > 0 and maximum_c < self.target_size[0] and maximum_r < self.target_size[1]:
                indices += [index]

        return indices

    @staticmethod
    def _crop_bounding_boxes(bounding_boxes, boundary):
        cropped_bounding_boxes = numpy.array(boundary)

        bounding_boxes = bounding_boxes.copy()

        bounding_boxes[..., :2] = numpy.maximum(bounding_boxes[..., :2], cropped_bounding_boxes[:2])
        bounding_boxes[..., 2:] = numpy.minimum(bounding_boxes[..., 2:], cropped_bounding_boxes[2:])

        bounding_boxes[..., :2] -= cropped_bounding_boxes[:2]
        bounding_boxes[..., 2:] -= cropped_bounding_boxes[:2]

        mask = numpy.all(bounding_boxes[..., :2] < bounding_boxes[..., 2:], axis=2)

        bounding_boxes[~mask] = numpy.zeros((4,))

        return bounding_boxes

    def _crop_image(self, image):
        crop_r = numpy.random.randint(0, image.shape[0] - self.generator.crop_size[0] - 1)
        crop_c = numpy.random.randint(0, image.shape[1] - self.generator.crop_size[1] - 1)

        crop = image[
               crop_r:crop_r + self.generator.crop_size[0],
               crop_c:crop_c + self.generator.crop_size[1],
               ...
               ]

        dimensions = numpy.array([
            crop_r,
            crop_c,
            crop_r + self.generator.crop_size[0],
            crop_c + self.generator.crop_size[1]
        ])

        return crop, dimensions

    def _get_batches_of_transformed_samples(self, selection):
        # TODO: permit batch sizes > 1
        batch_index, image_index = 0, selection[0]

        while True:
            try:
                x = self._transform_samples(batch_index, image_index)
            except BoundingBoxException:
                # FIXME: This should do something! To ignore the image
                image_index += 1
                continue
            except MissingImageException:
                image_index = 0
                continue
            break

        return x, None

    def _transform_samples(self, batch_index, image_index):
        x_bounding_boxes = numpy.zeros(
            (self.batch_size, 0, 4)
        )

        x_categories = numpy.zeros(
            (self.batch_size, 0, self.n_categories)
        )

        x_images = numpy.zeros(
            (self.batch_size, *self.target_size, self.channels)
        )

        x_masks = numpy.zeros(
            (self.batch_size, 0, *self.mask_size)
        )

        x_metadata = numpy.zeros(
            (self.batch_size, 3)
        )

        horizontal_flip = False

        if self.generator.horizontal_flip:
            if numpy.random.random() < 0.5:
                horizontal_flip = True

        vertical_flip = False

        if self.generator.vertical_flip:
            if numpy.random.random() < 0.5:
                vertical_flip = True

        try:
            pathname = self.dictionary[image_index]["image"]["pathname"]

        except:
            raise MissingImageException

        target_image = numpy.zeros((*self.target_size, self.channels))

        image = skimage.io.imread(pathname)

        if self.data_format == "channels_last":
            image = image[..., :self.channels]
        else:
            image = image[:self.channels, ...]

        dimensions = numpy.array([0, 0, image.shape[0], image.shape[1]])

        if self.generator.crop_size:
            if image.shape[0] > self.generator.crop_size[0] and image.shape[1] > self.generator.crop_size[1]:
                image, dimensions = self._crop_image(image)

        dimensions = dimensions.astype(numpy.float16)

        scale = self.find_scale(image)

        dimensions *= scale

        image = skimage.transform.rescale(
            image,
            scale,
            anti_aliasing=True,
            mode="reflect",
            multichannel=True
        )

        image = self.generator.standardize(image)

        if horizontal_flip:
            image = numpy.fliplr(image)

        if vertical_flip:
            image = numpy.flipud(image)

        image_r = image.shape[0]
        image_c = image.shape[1]

        target_image[:image_r, :image_c] = image

        x_images[batch_index] = target_image

        x_metadata[batch_index] = [*self.target_size, 1.0]

        bounding_boxes = self.dictionary[image_index]["objects"]

        n_objects = len(bounding_boxes)

        if n_objects == 0:
            return [
                numpy.zeros((self.batch_size, 0, 4)),
                numpy.zeros((self.batch_size, 0, self.n_categories)),
                x_images,
                numpy.zeros((self.batch_size, 0, self.mask_size[0], self.mask_size[1])),
                x_metadata
            ]

        x_bounding_boxes = numpy.resize(
            x_bounding_boxes, (self.batch_size, n_objects, 4)
        )

        x_masks = numpy.resize(
            x_masks, (self.batch_size, n_objects, *self.mask_size)
        )

        x_categories = numpy.resize(
            x_categories,
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

            x_bounding_boxes[
                batch_index,
                bounding_box_index
            ] = target_bounding_box

            if "mask" in bounding_box:
                target_mask = skimage.io.imread(
                    bounding_box["mask"]["pathname"]
                )

                target_mask = skimage.transform.rescale(
                    target_mask,
                    scale,
                    anti_aliasing=True,
                    mode="reflect",
                    multichannel=False
                )

                target_mask = target_mask[minimum_r:maximum_r + 1, minimum_c:maximum_c + 1]

                target_mask = skimage.transform.resize(
                    target_mask,
                    self.mask_size,
                    order=0,
                    mode="reflect",
                    anti_aliasing=True
                )

                x_masks[
                    batch_index,
                    bounding_box_index
                ] = target_mask

            target_category = numpy.zeros(self.n_categories)

            target_category[self.categories[bounding_box["category"]]] = 1

            x_categories[
                batch_index,
                bounding_box_index
            ] = target_category

        x = self._shuffle_objects(x_bounding_boxes, x_categories, x_masks)

        x_bounding_boxes, x_categories, x_masks = x

        x_bounding_boxes = self._crop_bounding_boxes(x_bounding_boxes, dimensions)

        cropped = self._cropped_objects(x_bounding_boxes)

        x_bounding_boxes = x_bounding_boxes[:, ~cropped]

        x_categories = x_categories[:, ~cropped]

        x_masks = x_masks[:, ~cropped]

        for bounding_box_index, bounding_box in enumerate(x_bounding_boxes[0]):
            mask = x_masks[0, bounding_box_index]

            if horizontal_flip:
                bounding_box = [
                    bounding_box[0],
                    image.shape[1] - bounding_box[3],
                    bounding_box[2],
                    image.shape[1] - bounding_box[1]
                ]
                mask = numpy.fliplr(mask)

            if vertical_flip:
                bounding_box = [
                    image.shape[0] - bounding_box[2],
                    bounding_box[1],
                    image.shape[0] - bounding_box[0],
                    bounding_box[3]
                ]
                mask = numpy.flipud(mask)

            x_bounding_boxes[
                batch_index,
                bounding_box_index
            ] = bounding_box

            x_masks[
                batch_index,
                bounding_box_index
            ] = mask

        if self.generator.clear_border:
            indices = self._clear_border(x_bounding_boxes)

            x_bounding_boxes = x_bounding_boxes[:, indices]

            x_categories = x_categories[:, indices]

            x_masks = x_masks[:, indices]

        if x_bounding_boxes.shape == (self.batch_size, 0, 4):
            raise BoundingBoxException

        x_masks[x_masks > 0.5] = 1.0
        x_masks[x_masks < 0.5] = 0.0

        return [
            x_bounding_boxes,
            x_categories,
            x_images,
            x_masks,
            x_metadata
        ]


    @staticmethod
    def _cropped_objects(x_bounding_boxes):
        return numpy.all(x_bounding_boxes[..., :] == 0, axis=2)[0]

    def _shuffle_objects(self, x_bounding_boxes, x_categories, x_masks):
        n = x_bounding_boxes.shape[1]

        if self.shuffle:
            indicies = numpy.random.permutation(n)
        else:
            indicies = numpy.arange(0, n)

        return [
            x_bounding_boxes[:, indicies],
            x_categories[:, indicies],
            x_masks[:, indicies]
        ]


class ObjectDetectionGenerator:
    def __init__(
            self,
            clear_border=False,
            crop_size=None,
            data_format=None,
            horizontal_flip=False,
            preprocessing_function=None,
            rescale=False,
            rotation_range=0.0,
            samplewise_center=False,
            vertical_flip=False
    ):
        self.clear_border = clear_border

        self.crop_size = crop_size

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
        image = skimage.exposure.rescale_intensity(image, out_range=(0.0, 1.0))

        if self.preprocessing_function:
            image = self.preprocessing_function(image)

        if self.rescale:
            image *= self.rescale

        if self.samplewise_center:
            image -= numpy.mean(image, keepdims=True)

        return image