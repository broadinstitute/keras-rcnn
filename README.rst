Keras-RCNN
==========

.. image:: https://travis-ci.org/broadinstitute/keras-rcnn.svg?branch=master
    :target: https://travis-ci.org/broadinstitute/keras-rcnn

.. image:: https://codecov.io/gh/broadinstitute/keras-rcnn/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/broadinstitute/keras-rcnn

keras-rcnn is *the* Keras package for region-based convolutional
neural networks.

Requirements
---------------
Python 3

keras-resnet==0.2.0

numpy==1.16.2

tensorflow==1.13.1

Keras==2.2.4

scikit-image==0.15.0


Getting Started
---------------

Let’s read and inspect some data:

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

    target, _ = generator.next()
    
    target_bounding_boxes, target_categories, target_images, target_masks, target_metadata = target

    target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

    target_images = numpy.squeeze(target_images)

    target_categories = numpy.argmax(target_categories, -1)

    target_categories = numpy.squeeze(target_categories)

    keras_rcnn.utils.show_bounding_boxes(target_images, target_bounding_boxes, target_categories)


Let’s create an RCNN instance:

.. code:: python

    model = keras_rcnn.models.RCNN((224, 224, 3), ["circle", "rectangle", "triangle"])

and pass our preferred optimizer to the `compile` method:

.. code:: python

    optimizer = keras.optimizers.Adam(0.0001)

    model.compile(optimizer)

Finally, let’s use the `fit_generator` method to train our network:

.. code:: python

    model.fit_generator(    
        epochs=10,
        generator=generator,
        validation_data=validation_data
    )

External Data
-------------

The data is made up of a list of dictionaries corresponding to images. 

* For each image, add a dictionary with keys 'image', 'objects'
    * 'image' is a dictionary, which contains keys 'checksum', 'pathname', and 'shape'
        * 'checksum' is the md5 checksum of the image
        * 'pathname' is the pathname of the image, put in full pathname
        * 'shape' is a dictionary with keys 'r', 'c', and 'channels'
            * 'c': number of columns
            * 'r': number of rows
            * 'channels': number of channels
    * 'objects' is a list of dictionaries, where each dictionary has keys 'bounding_box', 'category'
        * 'bounding_box' is a dictionary with keys 'minimum' and 'maximum'
            * 'minimum': dictionary with keys 'r' and 'c'
                * 'r': smallest bounding box row
                * 'c': smallest bounding box column
            * 'maximum': dictionary with keys 'r' and 'c'
                * 'r': largest bounding box row
                * 'c': largest bounding box column
        * 'category' is a string denoting the class name

Suppose this data is save in a file called training.json. To load data,

.. code:: python

    import json

    with open('training.json') as f:
        d = json.load(f)


Slack
-----

We’ve been meeting in the #keras-rcnn channel on the keras.io Slack
server. 

You can join the server by inviting yourself from the following website:

https://keras-slack-autojoin.herokuapp.com/
