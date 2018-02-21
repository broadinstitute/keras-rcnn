Keras-RCNN
==========

.. image:: https://travis-ci.org/broadinstitute/keras-rcnn.svg?branch=master
    :target: https://travis-ci.org/broadinstitute/keras-rcnn

.. image:: https://codecov.io/gh/broadinstitute/keras-rcnn/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/broadinstitute/keras-rcnn

keras-rcnn is *the* Keras package for region-based convolutional
neural networks.

Getting Started
---------------

Let’s read:

.. code:: python

    training_dictionary, test_dictionary = keras_rcnn.datasets.shape.load_data()

    categories = {"circle": 1, "rectangle": 2, "triangle": 3}

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    generator = generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(224, 224)
    )

    validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    validation_data = validation_data.flow_from_dictionary(
        dictionary=test_dictionary,
        categories=categories,
        target_size=(224, 224)
    )

and inspect our training data:

.. code:: python

    x, _ = generator.next()
    
    target_bounding_boxes, target_categories, target_images, target_masks, target_metadata = x

    target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

    target_images = numpy.squeeze(target_images)

    target_categories = numpy.argmax(target_categories, -1)

    target_categories = numpy.squeeze(target_categories)

    _, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

    axis.imshow(target_images)

    for target_index, target_category in enumerate(target_categories):
        xy = [
            target_bounding_boxes[target_index][1],
            target_bounding_boxes[target_index][0]
        ]

        w = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]
        h = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]

        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

        axis.add_patch(rectangle)

    matplotlib.pyplot.show()


Let’s create an RCNN instance:

.. code:: python

    target_image = keras.layers.Input(
        shape=(224, 224, 3),
        name="target_image"
    )

    y = keras.applications.VGG19(include_top=False, input_tensor=target_image).layers[:-2]

and pass our preferred optimizer to the `compile` method:

.. code:: python

    optimizer = keras.optimizers.Adam(0.000001)

    model.compile(optimizer)

Finally, let’s use the `fit_generator` method to train our network:

.. code:: python

    model.fit_generator(    
        epochs=10,
        generator=generator,
        validation_data=validation_data
    )

Slack
-----

We’ve been meeting in the #keras-rcnn channel on the keras.io Slack
server. 

You can join the server by inviting yourself from the following website:

https://keras-slack-autojoin.herokuapp.com/
