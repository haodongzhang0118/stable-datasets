import tarfile
from PIL import Image
from ..utils import Dataset

import numpy as np
from tqdm import tqdm
import io


class Imagenette(Dataset):
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
    def classes(self):
        return {
            "n01440764": "tench",
            "n02102040": "springer",
            "n02979186": "casette_player",
            "n03000684": "chain_saw",
            "n03028079": "church",
            "n03394916": "French_horn",
            "n03417042": "garbage_truck",
            "n03425413": "gas_pump",
            "n03445777": "golf_ball",
            "n03888257": "parachute",
        }

    @property
    def md5(self):
        return {"imagenette2.tgz": "fe2fc210e6bb7c5664d602c3cd71e612"}

    @property
    def urls(self):
        return {
            "imagenette2.tgz": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        }

    @property
    def num_classes(self):
        return 10

    @property
    def label_to_name(self, label):
        return {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "sheep",
            9: "truck",
        }[label]

    def load(self):
        # Load train set
        train_images = list()
        train_labels = list()
        test_images = list()
        test_labels = list()

        tar = tarfile.open(self.path / self.name / "imagenette2.tgz", "r:gz")

        train_names = [
            name for name in tar.getnames() if "JPEG" in name and "train" in name
        ]
        test_names = [
            name for name in tar.getnames() if "JPEG" in name and "train" not in name
        ]

        for name in tqdm(train_names, ascii=True, desc="Loading training set"):
            image = tar.extractfile(name)
            image = image.read()
            image = Image.open(io.BytesIO(image)).convert("RGB")
            train_images.append(image)
            train_labels.append(name.split("/")[2])

        for name in tqdm(test_names, ascii=True, desc="Loading test set"):
            image = tar.extractfile(name)
            image = image.read()
            image = Image.open(io.BytesIO(image)).convert("RGB")
            test_images.append(image)
            test_labels.append(name.split("/")[2])

        train_labels = (
            np.unique(train_labels) == np.array(train_labels)[:, None]
        ).argmax(1)
        test_labels = (np.unique(test_labels) == np.array(test_labels)[:, None]).argmax(
            1
        )

        self["train_X"] = train_images
        self["train_y"] = train_labels
        self["test_X"] = test_images
        self["test_y"] = test_labels
        return self
