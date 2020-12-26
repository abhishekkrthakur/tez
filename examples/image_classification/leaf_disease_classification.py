# For dataset, go to: https://www.kaggle.com/c/cassava-leaf-disease-classification
# For train_folds, go to: https://www.kaggle.com/abhishek/cassava-train-folds/
import argparse
import os

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

INPUT_PATH = "../input/"
IMAGE_PATH = "../input/train_images/"
MODEL_PATH = "../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
EPOCHS = 20
IMAGE_SIZE = 256


class LeafModel(tez.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained("efficientnet-b3")
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1536, num_classes)
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None


if __name__ == "__main__":
    train_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE),
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
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5),
        ],
        p=1.0,
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    current_fold = int(args.fold)

    dfx = pd.read_csv(os.path.join(INPUT_PATH, "train_folds.csv"))
    df_train = dfx[dfx.kfold != current_fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == current_fold].reset_index(drop=True)

    train_image_paths = [os.path.join(IMAGE_PATH, x) for x in df_train.image_id.values]
    valid_image_paths = [os.path.join(IMAGE_PATH, x) for x in df_valid.image_id.values]
    train_targets = df_train.label.values
    valid_targets = df_valid.label.values

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

    model = LeafModel(num_classes=dfx.label.nunique())
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + f"_fold_{current_fold}.bin"),
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
    model.save(os.path.join(MODEL_PATH, MODEL_NAME + f"_fold_{current_fold}.bin"))
