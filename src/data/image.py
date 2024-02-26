import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from src.data.base import BaseDataModule
from src.data.ee_diffusiondata import EECelebHQDataset
from src.data.stereo import StereoImages
from src.utils.img_utils import get_image_size, get_image_size_cv2

from torch.utils.data import (
    ConcatDataset,
    Dataset,
    RandomSampler,
    Subset,
    random_split,
)

import lightning as pl
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class Cityscape(StereoImages):
    pass

class KittiStereo(StereoImages):
    pass

class KittiGeneral(StereoImages):
    pass
