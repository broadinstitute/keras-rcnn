import numpy

import keras_rcnn.preprocessing._object_detection
import keras_rcnn.datasets.dsb2018
import keras_rcnn.models
import keras
import skimage.io

class TestObjectDetectionGenerator:
    def test_flow_from_dictionary(self):
        classes = {
            "nucleus": 1
        }

        training, _ = keras_rcnn.datasets.dsb2018.load_data()

        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

        generator = generator.flow_from_dictionary(training, classes, target_size=(448, 448))

        generator.next()

        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator(data_format=None)

        generator.flow_from_dictionary(training, classes, color_mode="grayscale", target_size=(448, 448))

        # generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()
        #
        # generator.flow_from_dictionary(training, classes, color_mode="brg", target_size=(448, 448))
        


    def test_standardize(self):

        training, _ = keras_rcnn.datasets.dsb2018.load_data()

        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

        image = skimage.io.imread(training[0]['image']['pathname'])

        generator.standardize(image)

    def test_find_scale(self):
        pass

    def test_get_batches_of_transformed_samples(self):
        pass
