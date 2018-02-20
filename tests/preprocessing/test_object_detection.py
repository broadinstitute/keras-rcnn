import keras_rcnn.preprocessing


class TestObjectDetectionGenerator:
    def test_flow_from_dictionary(self, training_dictionary):
        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

        categories = {"circle": 1, "square": 2, "triangle": 3}

        generator = generator.flow_from_dictionary(
            dictionary=training_dictionary,
            categories=categories,
            target_size=(224, 224),
            shuffle=False
        )

        x, _ = generator.next()

        bounding_boxes, categories, images, masks, metadata = x

        assert bounding_boxes.shape == (1, 17, 4)

        assert categories.shape == (1, 17, 4)

        assert images.shape == (1, 224, 224, 3)

        assert masks.shape == (1, 17, 28, 28)

        assert metadata.shape == (1, 3)
