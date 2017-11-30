# -*- coding: utf-8 -*-

import keras.backend
import keras.preprocessing.image
import numpy
import skimage.io
import skimage.transform


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(
            self,
            dictionary,
            classes,
            target_size,
            generator,
            batch_size=1,
            crop_size=None,
            data_format=None,
            seed=None,
            shuffle=False,
            horizontal_flip=False,
            vertical_flip=False
    ):
        # We assume the user didn’t provide a background class so a
        # background label is pushed onto the “0” position of the classes
        # list:
        classes.insert(0, "background")

        # Create a class dictionary by zipping the classes with
        # corresponding indices, e.g.
        #
        #   {
        #       "background": 0,
        #       "aeroplane": 1,
        #       "bicycle": 2,
        #       …
        #   }
        indices = [index for index in range(0, len(classes))]

        self.classes = dict(zip(classes, indices))

        # Count the number of images in the dictionary:
        self.count = len(dictionary)

        self.channels = 3

        self.dictionary = dictionary

        self.generator = generator

        # Compute the minimum and maximum dimensions:
        rr, cc = target_size

        shapes = sum([image["shape"][:2] for image in dictionary], [])

        self.maximum = max(max(shapes), max([rr, cc]))

        self.minimum = max(min(shapes), min([rr, cc]))

        self.crop_size = crop_size

        self.horizontal_flip = horizontal_flip

        self.vertical_flip = vertical_flip

        self.target_size = target_size

        args = [self.count, batch_size, shuffle, seed]

        super(DictionaryIterator, self).__init__(*args)

    def rescale(self, image):
        rr, cc, _ = image.shape

        scale = self.minimum / numpy.min([rr, cc])

        maximum_side = numpy.max([rr, cc])

        if maximum_side * scale > self.maximum:
            scale = self.maximum / maximum_side

        image = skimage.transform.rescale(image, scale, mode="reflect")

        return image, scale

    def next(self):
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        bounding_boxes = numpy.zeros((batch_size, 0, 4))

        classes = numpy.zeros((batch_size, 0))

        images = numpy.zeros((batch_size, *self.target_size, self.channels))

        metadata = numpy.zeros((batch_size, 3))

        for batch_index, image_index in enumerate(selection):
            filename = self.dictionary[image_index]["filename"]

            # TODO: What happens when the file doesn't exist?
            image = skimage.io.imread(filename)

            # if self.crop_size:
            #     rr_crop, cc_crop = self.crop_size
            #
            #     x1_crop = numpy.random.randint(0, image.shape[0] - rr_crop)
            #     y1_crop = numpy.random.randint(0, image.shape[1] - cc_crop)
            #
            #     image = image[x1_crop:x1_crop+rr_crop, y1_crop:y1_crop+cc_crop]

            image, scale = self.rescale(image)

            objects = self.dictionary[image_index]["boxes"]

            bounding_boxes, classes, image = self.generator.random_transform(image, objects)

            counter = []

            # if self.crop_size:
            #     cropped_bounding_boxes = []
            #
            #     for index,(x1, y1, x2, y2) in enumerate(bounding_boxes):
            #         x1 = x1 - x1_crop
            #         x2 = x2 - x1_crop
            #         y1 = y1 - y1_crop
            #         y2 = y2 - y1_crop
            #
            #         if x1 >= 0 and x2 < cc_crop and y1 >= 0 and y2 < rr_crop:
            #             bounding_box = [x1, y1, x2, y2]
            #
            #             cropped_bounding_boxes.append(bounding_box)
            #             counter.append(index)
            #
            #     bounding_boxes = cropped_bounding_boxes

            bounding_boxes = numpy.array(bounding_boxes, numpy.float64)

            # FIXME: This only works when batch_size is 1.
            bounding_boxes = numpy.expand_dims(bounding_boxes, 0)

            bounding_boxes[batch_index] *= scale

            # Pad the rescaled image with zeros.
            rr, cc, channels = image.shape

            images[batch_index, :rr, :cc, :channels] = image

            metadata[batch_index] = [rr, cc, scale]

            classes = [self.classes[class_index] for class_index in classes]

            if self.crop_size:
                classes = [classes[index] for index in counter]

            classes = numpy.array(classes)

            classes = keras.utils.to_categorical(classes, len(self.classes))

            # FIXME: This only works when batch_size is 1.
            classes = numpy.expand_dims(classes, 0)

        return [bounding_boxes, images, classes, metadata], None


class ImageSegmentationGenerator:
    def flow(
            self,
            dictionary,
            classes,
            target_size,
            batch_size=1,
            seed=None,
            shuffle=True
    ):
        return DictionaryIterator(
            batch_size=batch_size,
            classes=classes,
            dictionary=dictionary,
            generator=self,
            seed=seed,
            shuffle=shuffle,
            target_size=target_size
        )


class ObjectDetectionGenerator:
    def __init__(
            self,
            crop_size=None,
            data_format=None,
            horizontal_flip=False,
            vertical_flip=False
    ):
        if data_format is None:
            data_format = keras.backend.image_data_format()

        self.data_format = data_format

        if data_format == "channels_first":
            self.channel_axis = 1
            self.rr_axis = 2
            self.cc_axis = 3

        if data_format == "channels_last":
            self.channel_axis = 3
            self.rr_axis = 1
            self.cc_axis = 2

        self.crop_size = crop_size

        self.horizontal_flip = horizontal_flip

        self.vertical_flip = vertical_flip

    def flow(
            self,
            dictionary,
            classes,
            target_size,
            batch_size=1,
            seed=None,
            shuffle=True
    ):
        return DictionaryIterator(
            batch_size=batch_size,
            classes=classes,
            crop_size=self.crop_size,
            dictionary=dictionary,
            generator=self,
            seed=seed,
            shuffle=shuffle,
            target_size=target_size,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            data_format=self.data_format
        )

    def random_transform(self, image, instances, seed=None):
        if seed is not None:
            numpy.random.seed(seed)

        rr, cc, channels = image.shape

        rr_axis = self.rr_axis - 1
        cc_axis = self.cc_axis - 1

        # ---

        horizontal_flip = False

        if self.horizontal_flip:
            if numpy.random.random() < 0.5:
                image = keras.preprocessing.image.flip_axis(image, cc_axis)

                horizontal_flip = True

        # ---

        vertical_flip = False

        if self.vertical_flip:
            if numpy.random.random() < 0.5:
                image = keras.preprocessing.image.flip_axis(image, rr_axis)

                vertical_flip = True

        # ---

        bounding_boxes = []

        classes = []

        for instance_index, instance in enumerate(instances):
            x1 = int(instance["x1"])
            x2 = int(instance["x2"])
            y1 = int(instance["y1"])
            y2 = int(instance["y2"])

            # If $x_maximum$ is equal to the image’s width, subtract
            # $x_maximum$ by 1.
            if x2 == image.shape[1]:
                x2 -= 1

            # Likewise, if $y_maximum$ is equal to the image’s height,
            # subtract $y_maximum$ by 1.
            if y2 == image.shape[0]:
                y2 -= 1

            if horizontal_flip:
                x1 = cc - x1
                x2 = cc - x2

            if vertical_flip:
                y1 = rr - y1
                y2 = rr - y2

            bounding_boxes.append([x1, y1, x2, y2])

            classes.append(instance["class"])

        return bounding_boxes, classes, image
