import matplotlib.cm
import matplotlib.patches
import matplotlib.pyplot
import numpy


def _get_cmap(n, name="hsv"):
    return matplotlib.cm.get_cmap(name, n)


# TODO: commit upstream to `scikit-image`.
def show_bounding_boxes(image, bounding_boxes, categories=None):
    axis = matplotlib.pyplot.gca()

    n = bounding_boxes.shape[0]

    if categories is not None:
        assert bounding_boxes.shape[0] == categories.shape[0]

        categories = categories.reshape((-1, 1))
    else:
        categories = numpy.zeros((bounding_boxes.shape[0], 1))

    axis.imshow(image)

    colormap = _get_cmap(n)

    for bounding_box, category in zip(bounding_boxes, categories):
        rectangle = matplotlib.patches.Rectangle(
            [
                bounding_box[1],
                bounding_box[0]
            ],
            bounding_box[3] - bounding_box[1],
            bounding_box[2] - bounding_box[0],
            edgecolor=colormap(category[0]),
            facecolor="none"
        )

        axis.add_patch(rectangle)
