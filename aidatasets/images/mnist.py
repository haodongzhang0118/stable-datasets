import os
import pickle
import gzip
from ..utils import Dataset
import time



class MNIST(Dataset):
    """

    The MNIST database of handwritten digits, available from this page
    has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image.

    It is a good database for people who want to try learning techniques
    and pattern recognition methods on real-world data while spending minimal
    efforts on preprocessing and formatting.

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


    @property
    def urls(self):
        return {"mnist.pkl.gz": "http://deeplearning.net/data/mnist/mnist.pkl.gz"}

    @property
    def image_shape(self):
        return (1,28,28)

    def load(self):
        t0 = time.time()
        print("Loading mnist")
        f = gzip.open(self.path / "mnist/mnist.pkl.gz", "rb")
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        f.close()
    
        self["train_X"] = train_set[0].reshape((-1, 28, 28, 1))
        self["train_y"] = train_set[1]
        self["test_X"] = test_set[0].reshape((-1, 28, 28, 1))
        self["test_y"] = test_set[1]
        self["valid_X"] = valid_set[0].reshape((-1, 28, 28, 1))
        self["valid_y"] = valid_set[1]
        print("Dataset mnist loaded in {0:.2f}s.".format(time.time() - t0))
        return dataset
