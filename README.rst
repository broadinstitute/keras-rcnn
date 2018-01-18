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

    training, validation, test = keras_rcnn.datasets.malaria_phenotypes.load_data()

    classes = {
        "rbc": 1, "leu": 2, "ring": 3, "tro": 4, "sch": 5, "gam": 6
    }

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    generator = generator.flow(training, classes)

    validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    validation_data = validation_data.flow(validation, classes)

and inspect our training data:

.. code:: python

    (target_bounding_boxes, target_image, target_scores, _), _ = generator.next()

    target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

    target_image = numpy.squeeze(target_image)

    target_scores = numpy.argmax(target_scores, -1)

    target_scores = numpy.squeeze(target_scores)

    _, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

    axis.imshow(target_image)

    for target_index, target_score in enumerate(target_scores):
        if target_score > 0:
            xy = [
                target_bounding_boxes[target_index][0],
                target_bounding_boxes[target_index][1]
            ]

            w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
            h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]

            rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

            axis.add_patch(rectangle)

    matplotlib.pyplot.show()

.. image:: https://storage.googleapis.com/keras-rcnn-website/example.png

Let’s create an RCNN instance:

.. code:: python

    image = keras.layers.Input((None, None, 3))

    model = keras_rcnn.models.RCNN(image, classes=len(classes) + 1)

and pass our preferred optimizer to the `compile` method:

.. code:: python

    optimizer = keras.optimizers.Adam(0.0001)

    model.compile(optimizer)

Finally, let’s use the `fit_generator` method to train our network:

.. code:: python

    model.fit_generator(generator)

Slack
-----

We’ve been meeting in the #keras-rcnn channel on the keras.io Slack
server. 

You can join the server by inviting yourself from the following website:

https://keras-slack-autojoin.herokuapp.com/
