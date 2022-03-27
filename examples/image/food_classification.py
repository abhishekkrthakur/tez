import argparse
import os

import albumentations
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from datasets import load_dataset
from joblib import Parallel, delayed
from sklearn import metrics, model_selection
from tqdm import tqdm

import tez
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.utils import seed_everything


if __name__ == "__main__":
    data = load_dataset("food101")
    print(data["train"][0])
