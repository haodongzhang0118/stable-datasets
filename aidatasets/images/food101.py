import os
import pickle
import tarfile
import time
from ..utils import download_dataset, Dataset
from PIL import Image

import numpy as np
from tqdm import tqdm



class Food101(Dataset):
    """Image classification.
    """


    @property
    def urls(self):
        return {
        "food-101.tar.gz": "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        }

    @property
    def num_classes(self):
        return 101

    @property
    def label_to_name(self, label):
        return self.loaded_names[label]
 
    @property
    def name(self):
        return "Food101"

    @property
    def webpage(self):
        return "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"

    @property
    def cite(self):
        return """@inproceedings{bossard14,
title = {Food-101 -- Mining Discriminative Components with Random Forests},
author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
booktitle = {European Conference on Computer Vision},
year = {2014}
}"""
    

    def load(self):
        t0 = time.time()
    
        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], 
                "r:gz")

        print("Loading labels")
        self.loaded_labels = tar.extractfile("food-101/meta/labels.txt").read()
        print("Loading train info")
        train = np.loadtxt(tar.extractfile("food-101/meta/train.txt"), dtype='str')
        print("Loading test info")
        test = np.loadtxt(tar.extractfile("food-101/meta/test.txt"), dtype='str')
        # Load train set
        train_images = list()
        train_labels = list()
        for name in tqdm(train, desc="Loading Train Food101", ascii=True):
            f = tar.extractfile(f"food-101/images/{name}.jpg")
            train_images.append(Image.open(f))
            train_labels.append(name.split("/")[0])

        # Load test set
        test_images = list()
        test_labels = list()
        for name in tqdm(test, desc="Loading Test Food101", ascii=True):
            f = tar.extractfile(f"food-101/images/{name}.jpg").read()
            test_images.append(Image.open(f))
            test_labels.append(name.split("/")[0])

   
        self["train_X"] =  np.transpose(train_images, (0, 2, 3, 1))
        self["train_y"] =  train_labels
        self["test_X"] =  np.transpose(test_images, (0, 2, 3, 1))
        self["test_y"] = test_labels
        print("Dataset cifar10 loaded in{0:.2f}s.".format(time.time() - t0))
