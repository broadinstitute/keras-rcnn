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
            target_shape,
            generator,
            batch_size=1,
            seed=None,
            shuffle=False,
    ):
        # We assume the user didn’t provide a background class so it’s
        # pushed onto the “0” position of the classes list:
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
        indices = [index for index in range(0, len(classes) + 1)]

        self.classes = dict(zip(classes, indices))

        self.count = len(dictionary)

        self.dictionary = dictionary

        self.generator = generator

        rr, cc, _ = target_shape

        shapes = sum([image["shape"][:2] for image in dictionary], [])

        self.maximum = max(max(shapes), max([rr, cc]))

        self.minimum = max(min(shapes), min([rr, cc]))

        self.target_shape = target_shape

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

        images = numpy.zeros((batch_size,) + self.target_shape)

        metadata = numpy.zeros((batch_size, 3))

        for batch_index, image_index in enumerate(selection):
            pathname = self.dictionary[image_index]["filename"]

            image = skimage.io.imread(pathname)

            image, scale = self.rescale(image)

            rr, cc, channels = image.shape

            images[batch_index, :rr, :cc, :channels] = image

            metadata[batch_index] = [rr, cc, scale]

            instances = self.dictionary[image_index]["boxes"]

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

                box = [x1, y1, x2, y2]

                bounding_boxes = numpy.append(bounding_boxes, [[box]], axis=1)

                class_index = self.classes[instance["class"]]

                classes = numpy.append(classes, [class_index])

            bounding_boxes[batch_index, :, :4] *= scale

            classes = keras.utils.to_categorical(classes, 22)

            classes = numpy.expand_dims(classes, 0)

        return [bounding_boxes, images, classes, metadata], None


class ImageSegmentationGenerator:
    def flow(self):
        pass


class ObjectDetectionGenerator:
    def flow(
            self,
            dictionary,
            classes,
            target_shape,
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
            target_shape=target_shape
        )
