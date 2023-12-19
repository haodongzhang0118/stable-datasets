import os
import pickle
import tarfile
import time
from ..utils import Dataset

import numpy as np
from tqdm import tqdm



class FGVCAircraft(Dataset):
    """Image classification.
    The `CIFAR-10 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.
    The dataset is divided into five training batches and one test batch,
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the
    remaining images in random order, but some training batches may
    contain more images from one class than another. Between them, the
    training batches contain exactly 5000 images from each class.

    Parameters
    ----------

    path: str (optional)
        default ($DATASET_PATH), the path to look for the data and
        where the data will be downloaded if not present

    Returns
    -------

    train_images: array

    train_labels: array

    test_images: array

    test_labels: array

    """


    @property
    def urls(self):
        return {
        "fgvc-aircraft-2013b.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
        }

    @property
    def num_classes(self):
        return 102
 
    @property
    def num_samples(self):
        return 10200

    @property
    def webpage(self):
        return "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"

    def load(self):
        t0 = time.time()
    
        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], 
                "r:gz")
    
        # Load train set
        train_images = list()
        train_labels = list()
        for k in tqdm(range(1, 6), desc="Loading cifar10", ascii=True):
            f = tar.extractfile("cifar-10-batches-py/data_batch_" + str(k)).read()
            data_dic = pickle.loads(f, encoding="latin1")
            train_images.append(data_dic["data"].reshape((-1, 3, 32, 32)))
            train_labels.append(data_dic["labels"])
        train_images = np.concatenate(train_images, 0)
        train_labels = np.concatenate(train_labels, 0)
    
        # Load test set
        f = tar.extractfile("cifar-10-batches-py/test_batch").read()
        data_dic = pickle.loads(f, encoding="latin1")
        test_images = data_dic["data"].reshape((-1, 3, 32, 32))
        test_labels = np.array(data_dic["labels"])
    
        self["train_X"] =  np.transpose(train_images, (0, 2, 3, 1))
        self["train_y"] =  train_labels
        self["test_X"] =  np.transpose(test_images, (0, 2, 3, 1))
        self["test_y"] = test_labels
        print("Dataset cifar10 loaded in{0:.2f}s.".format(time.time() - t0))
