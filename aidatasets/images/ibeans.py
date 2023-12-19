#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import time
import zipfile
import urllib
import numpy as np
from ..utils import Dataset
from PIL import Image


class IBeans(Dataset):
    """Plant images classification.

    This dataset is of leaf images taken in the field in different
    districts in Uganda by the Makerere AI lab in collaboration with the
    National Crops Resources Research Institute (NaCRRI), the national
    body in charge of research in agriculture in Uganda.

    The goal is to build a robust machine learning model that is able
    to distinguish between diseases in the Bean plants. Beans are an
    important cereal food crop for Africa grown by many small-holder
    farmers - they are a significant source of proteins for school-age
    going children in East Africa.

    The data is of leaf images representing 3 classes: the healthy class of
    images, and two disease classes including Angular Leaf Spot and Bean
    Rust diseases. The model should be able to distinguish between these 3
    classes with high accuracy. The end goal is to build a robust, model
    that can be deployed on a mobile device and used in the field by a
    farmer.

    The data includes leaf images taken in the field. The figure above
    depicts examples of the types of images per class. Images were taken
    from the field/garden a basic smartphone.

    The images were then annotated by experts from NaCRRI who determined
    for each image which disease was manifested. The experts were part of
    the data collection team and images were annotated directly during the
    data collection process in the field.

    Class   Examples
    Healthy class   428
    Angular Leaf Spot   432
    Bean Rust   436
    Total:  1,296

    Data Released   20-January-2020
    License     MIT
    Credits     Makerere AI Lab

    Parameters
    ----------
        path: str (optional)
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        valid_images: array

        valid_labels: array

        test_images: array

        test_labels: array

    """
 
    def classes(self):
        return ["angular_leaf_spot", "bean_rust", "healthy"]

    @property
    def urls(self):
        return {
                "train.zip": "https://storage.googleapis.com/ibeans/train.zip",
                "test.zip":"https://storage.googleapis.com/ibeans/test.zip",
                "validation.zip":"https://storage.googleapis.com/ibeans/validation.zip"
        }

    
    def load(self):
        t0 = time.time()
    
        # Loading the file
        train_images = list()
        train_labels = list()
        f = zipfile.ZipFile(path + "ibeans/train.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            train_images.append(Image.open(io.BytesIO(f.read(filename))))
            train_labels.append(ibeans.classes.index(filename.split("/")[1]))
    
        # Loading the file
        test_images = list()
        test_labels = list()
        f = zipfile.ZipFile(path + "ibeans/test.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            test_images.append(Image.open(io.BytesIO(f.read(filename))))
            test_labels.append(ibeans.classes.index(filename.split("/")[1]))
    
        # Loading the file
        valid_images = list()
        valid_labels = list()
        f = zipfile.ZipFile(path + "ibeans/validation.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            valid_images.append(Image.open(io.BytesIO(f.read(filename))))
            valid_labels.append(ibeans.classes.index(filename.split("/")[1]))
   
        self["train_set/images"] =np.array(train_images)
        self["test_set/images"]= np.array(test_images)
        self["valid_set/images"]= np.array(valid_images)
        self["train_set/labels"]= np.array(train_labels)
        self["test_set/labels"]= np.array(test_labels)
        self["valid_set/labels"]= np.array(valid_labels)
    
        print("Dataset ibeans loaded in {0:.2f}s.".format(time.time() - t0))
    
