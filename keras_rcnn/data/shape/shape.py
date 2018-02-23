import hashlib
import json
import os.path
import shutil
import uuid

import skimage.draw
import skimage.io


def md5sum(pathname, blocksize=65536):
    checksum = hashlib.md5()

    with open(pathname, "rb") as stream:
        for block in iter(lambda: stream.read(blocksize), b""):
            checksum.update(block)

    return checksum.hexdigest()


def __main__():
    pathname = "images"

    if os.path.exists(pathname): shutil.rmtree(pathname)

    os.mkdir(pathname)

    groups = ("training", "test")

    r, c = 224, 224

    for group in groups:
        dictionaries = []

        for _ in range(256):
            identifier = uuid.uuid4()

            image, objects = skimage.draw.random_shapes((r, c), 32, 2, 32)

            filename = "{}.png".format(identifier)

            pathname = os.path.join("images", filename)

            skimage.io.imsave(pathname, image)

            if os.path.exists(pathname):
                dictionary = {
                    "image": {
                        "checksum": md5sum(pathname),
                        "pathname": pathname,
                        "shape": {
                            "r": r,
                            "c": c,
                            "channels": 3
                        }
                    },
                    "objects": []
                }

                for category, (bounding_box_r, bounding_box_c) in objects:
                    minimum_r, maximum_r = bounding_box_r
                    minimum_c, maximum_c = bounding_box_c

                    object_dictionary = {
                        "bounding_box": {
                            "minimum": {
                                "r": minimum_r - 1,
                                "c": minimum_c - 1
                            },
                            "maximum": {
                                "r": maximum_r - 1,
                                "c": maximum_c - 1
                            }
                        },
                        "category": category
                    }

                    dictionary["objects"].append(object_dictionary)

                dictionaries.append(dictionary)

        filename = "{}.json".format(group)

        with open(filename, "w") as stream:
            json.dump(dictionaries, stream)


if __name__ == "__main__":
    __main__()
