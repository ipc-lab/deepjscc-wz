import os
from random import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import lightning as pl
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from lightning import LightningDataModule, seed_everything
from sklearn.utils import shuffle
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    RandomSampler,
    Subset,
    random_split,
)
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.data.base import BaseDataModule


class StereoImages(BaseDataModule):
    def __init__(
        self,
        name: str = "Cityscape",  # KittiGeneral, Cityscape, KittiStereo
        resize=(128, 256),
        **kwargs
    ):
        super().__init__(name=name, resize=resize, **kwargs)

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.num_classes = -1
        self.input_dims = (3, resize[0], resize[1])

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = PairKittiCityscape(
                self.hparams.data_dir, self.hparams.name, "train", self.hparams.resize
            )
            self.data_val = PairKittiCityscape(
                self.hparams.data_dir, self.hparams.name, "val", self.hparams.resize
            )
            self.data_test = PairKittiCityscape(
                self.hparams.data_dir, self.hparams.name, "test", self.hparams.resize
            )


# Adapted from https://github.com/ipc-lab/NDIC-CAM/blob/main/dataset/PairKitti.py
class PairKittiCityscape(Dataset):
    def __init__(self, path, data, set_type, resize=(128, 256)):
        super(Dataset, self).__init__()
        self.resize = resize
        self.ar = []
        self.data = data

        if data == "KittiStereo":
            idx_path = os.path.join(path, "KITTI_stereo_" + set_type + ".txt")
        elif data == "KittiGeneral":
            idx_path = os.path.join(path, "KITTI_general_" + set_type + ".txt")
        elif data == "Cityscape":
            path = os.path.join(path, "cityscape_dataset")
            idx_path = os.path.join(path, "cityscape_" + set_type + ".txt")

        # pkl_path = idx_path.replace("cityscape_dataset/", "").replace(".txt", ".pth")

        # if os.path.exists(pkl_path):
        #    self.images = torch.load(pkl_path)
        # else:
        if data == "Cityscape":
            self.dataset = {
                "left": os.path.join(path, "leftImg8bit", set_type),
                "right": os.path.join(path, "rightImg8bit", set_type),
            }

            self.cities = [
                item
                for item in os.listdir(self.dataset["left"])
                if os.path.isdir(os.path.join(self.dataset["left"], item))
            ]

            self.ar = []
            for city in self.cities:
                pair_names = [
                    "_".join(f.split("_")[:-1])
                    for f in os.listdir(os.path.join(self.dataset["left"], city))
                    if os.path.splitext(f)[-1].lower() == ".png"
                ]
                for pair in pair_names:
                    left_img = os.path.join(self.dataset["left"], city, pair + "_leftImg8bit.png")
                    right_img = os.path.join(
                        self.dataset["right"], city, pair + "_rightImg8bit.png"
                    )
                    self.ar.append((left_img, right_img))
        else:
            with open(idx_path) as f:
                content = f.readlines()

            for i in range(0, len(content), 2):
                left_id = content[i].strip()
                right_id = content[i + 1].strip()
                self.ar.append((os.path.join(path, left_id), os.path.join(path, right_id)))
                # if set_type == "train":
                #    self.ar.append((path + '/' + right_id, path + '/' + left_id))
        """
        self.images = []
        for index in range(len(self.ar)):
            self.images.append(self.get_item(index))
        """
        # torch.save(self.images, pkl_path)

        self.images = {}

    def __getitem__(self, index):
        if index not in self.images:
            self.images[index] = self.get_item(index)  # self.images[index]

        return self.images[index]

    def transform(self, img):
        # Center Crop
        if self.data in ["KittiStereo", "KittiGeneral"]:
            img = TF.center_crop(img, (370, 740))

        img = TF.resize(img, self.resize)

        # Random Horizontal Flip
        # if random() > 0.5:
        #    img = TF.hflip(img)

        img = transforms.ToTensor()(img)

        return img

    def get_item(self, index):
        left_path, right_path = self.ar[index]

        left_img = self.transform(Image.open(left_path))
        right_img = self.transform(Image.open(right_path))

        return (left_img, right_img), -1

    def __len__(self):
        return len(self.ar)

    def __str__(self):
        return self.data
