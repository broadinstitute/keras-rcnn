import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    extras_require={
        "dev": ["black>=19.10b0", "check-manifest>=0.41", "pre-commit>=2.2.0"],
        "test": [
            "codecov",
            "mock",
            "pytest",
            "pytest-cov",
            "pytest-pep8",
            "pytest-runner",
        ],
    },
    install_requires=[
        "jsonschema>=3.2.0",
        "keras>=2.3.1",
        "keras-resnet>=0.2.0",
        "scikit-image>=0.17.2",
    ],
    license="MIT",
    name="keras-rcnn",
    package_data={
        "keras-rcnn": [
            "data/checkpoints/*/*.hdf5",
            "data/logs/*/*.csv",
            "data/notebooks/*/*.ipynb",
        ]
    },
    packages=setuptools.find_packages(exclude=["tests"]),
    url="https://github.com/broadinstitute/keras-rcnn",
    version="0.1.0",
    zip_safe=True,
)
