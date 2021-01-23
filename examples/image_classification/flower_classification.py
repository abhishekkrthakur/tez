# For dataset, go to: https://www.kaggle.com/msheriey/104-flowers-garden-of-eden
# Update INPUT_PATH and MODEL_PATH
import argparse
import os

import glob
import albumentations
import pandas as pd
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
from torch.nn import functional as F

INPUT_PATH = "../../input/"
MODEL_PATH = "../../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 20


class FlowerModel(tez.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained("efficientnet-b0")
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1280, num_classes)

    def monitor_metrics(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, 0, {}


if __name__ == "__main__":
    train_aug = albumentations.Compose(
        [
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    train_image_paths = glob.glob(
        os.path.join(
            INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "train", "**", "*.jpeg"
        ),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(
            INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "val", "**", "*.jpeg"
        ),
        recursive=True,
    )

    train_targets = [x.split("/")[-2] for x in train_image_paths]
    valid_targets = [x.split("/")[-2] for x in valid_image_paths]

    lbl_enc = preprocessing.LabelEncoder()
    train_targets = lbl_enc.fit_transform(train_targets)
    valid_targets = lbl_enc.transform(valid_targets)

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=train_aug,
    )

    valid_dataset = ImageDataset(
        image_paths=valid_image_paths,
        targets=valid_targets,
        augmentations=valid_aug,
    )

    model = FlowerModel(num_classes=len(lbl_enc.classes_))

    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
        patience=3,
        mode="min",
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=TRAIN_BATCH_SIZE,
        valid_bs=VALID_BATCH_SIZE,
        device="cuda",
        epochs=EPOCHS,
        callbacks=[es],
        fp16=True,
    )
