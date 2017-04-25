import keras.layers
import keras.models
import keras_rcnn.models


class TestRPN:
    def test_constructor(self):
        shape = (256, 256, 3)

        image = keras.layers.Input(shape)

        trunk = keras_rcnn.models.nn_base(image)

        anchors = 9

        region_proposal_network = keras_rcnn.models.rpn(trunk, anchors)

        x, y = image, region_proposal_network[:2]

        region_proposal_network = keras.models.Model(x, y)

        import IPython
        IPython.embed()

        assert True
