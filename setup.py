import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    extras_require={
        "test": [
            "codecov",
            "mock",
            "pytest",
            "pytest-cov",
            "pytest-pep8",
            "pytest-runner"
        ],
    },
    install_requires=[
        "jsonschema",
        "keras",
        "keras-resnet",
        "scikit-image"
    ],
    license="MIT",
    name="keras-rcnn",
    package_data={
        "keras-rcnn": [
            "data/checkpoints/*/*.hdf5",
            "data/logs/*/*.csv",
            "data/notebooks/*/*.ipynb"
        ]
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/keras-rcnn",
    version="0.0.2",
    zip_safe=True
)
