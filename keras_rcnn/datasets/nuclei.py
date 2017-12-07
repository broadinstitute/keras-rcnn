# -*- coding: utf-8 -*-
import keras_rcnn.datasets
'''
Load Broad Institute’s nuclei dataset.

Broad Institute’s nuclei dataset is a collection of images from the BBBC022 
data set of 100 manually annotated images (~10,000 nuclei) taken from a 
compound profiling experiment of human U2OS cells. 
The cells were stained with Hoechst 33342 to highlight their DNA.
'''
def load_data():
    return keras_rcnn.datasets.load_data("nuclei")
