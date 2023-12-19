import os
from ..utils import Dataset
import time
import zipfile
import imageio
from tqdm import tqdm
import numpy as np


class RockPaperScissor(Dataset):

    @property
    def urls(self):
        return {
                "rps.zip":"https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
         "rps-test-set.zip":"https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip",
}

    def load(self):
   
    
        t0 = time.time()
    
        # Loading the file
        print("Loading mnist")
        test_images = []
        test_classes = []
        test_styles = []
        train_images = []
        train_classes = []
        train_styles = []
        with zipfile.ZipFile(self.path / self.name/ "rps-test-set.zip", "r") as zfile:
            for filename in tqdm(zfile.namelist(), desc="test set", ascii=True):
                if ".png" not in filename:
                    continue
                test_classes.append(filename.split("/")[1])
                test_styles.append(filename.split("-")[-2][-2:])
                test_images.append(imageio.imread(zfile.read(filename)))
    
        with zipfile.ZipFile(self.path / self.name / "rps.zip", "r") as zfile:
            for filename in tqdm(zfile.namelist(), desc="train set", ascii=True):
                if ".png" not in filename:
                    continue
                train_classes.append(filename.split("/")[1])
                train_styles.append(filename.split("-")[0][-2:])
                train_images.append(imageio.imread(zfile.read(filename)))
    
        self["train_X"] = np.array(train_images)
        self["train_y"] = np.array(train_classes)
        self["train_style"] = np.array(train_styles)
        self["test_X"] = np.array(test_images)
        self["test_y"] = np.array(test_classes)
        self["test_style"] = np.array(test_styles)

        print("Dataset rps loaded in {0:.2f}s.".format(time.time() - t0))
    
