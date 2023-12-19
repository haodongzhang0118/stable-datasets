import os
import pickle
import tarfile
import time
from ..utils import Dataset
import scipy.io
import numpy as np
from PIL import Image

class Flowers102(Dataset):
    """Image classification.

    We have created a 102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The details of the categories and the number of images for each class can be found on this category statistics page.

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features. """

    @property
    def website(self):
        return "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"

    @property
    def urls(self):
        return {
                "102flowers.tgz": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
                "imagelabels.mat":"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
                "setid.mat":"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
                "102segmentations.tgz":"https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz"
                }

    @property
    def num_classes(self):
        return 102
    
    def load(self):
    
        t0 = time.time()
        # Loading the file
        labels = scipy.io.loadmat(self.path / self.name / "imagelabels.mat")["labels"][0]
        ids = scipy.io.loadmat(self.path / self.name / "setid.mat")
        train_ids = ids["trnid"][0]
        test_ids = ids["tstid"][0]
        valid_ids = ids["valid"][0]

        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], "r:gz")
        train_images = []
        for name in tar.getnames():
            f = tar.extractfile(name)
            train_images.append(Image.open(f))
            print(name)
            asdf

    
        self["test_X"] = np.transpose(test_images, (0, 2, 3, 1))
        self["test_y"] = test_fine
        self["test_y_coarse"] = test_coarse
        print("Dataset cifar100 loaded in {0:.2f}s.".format(time.time() - t0))
