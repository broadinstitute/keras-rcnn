class TestObjectDetectionGenerator:
    def test_flow_from_dictionary(self, generator):
        x, _ = generator.next()

        bounding_boxes, categories, images, masks, metadata = x

        assert bounding_boxes.shape == (1, 17, 4)

        assert categories.shape == (1, 17, 4)

        assert images.shape == (1, 224, 224, 3)

        assert masks.shape == (1, 17, 28, 28)

        assert metadata.shape == (1, 3)
