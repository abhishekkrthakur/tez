import glob
import os

import albumentations
import timm
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
from torch.nn import functional as F

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
import numpy as np
import pandas as pd


INPUT_PATH = "../../input/"
MODEL_PATH = "../../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 20


class FlowerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        f1 = metrics.f1_score(targets, outputs, average="macro")
        return {"f1": torch.tensor(f1, device=device)}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        outputs = self.model(image)
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
            albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
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

    test_aug = albumentations.Compose(
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
        os.path.join(INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "train", "**", "*.jpeg"),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "val", "**", "*.jpeg"),
        recursive=True,
    )

    test_image_paths = glob.glob(
        os.path.join(INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "test", "*.jpeg"),
    )

    train_targets = [x.split("/")[-2] for x in train_image_paths]
    valid_targets = [x.split("/")[-2] for x in valid_image_paths]

    train_targets = [CLASSES[c] for c in train_targets]
    valid_targets = [CLASSES[c] for c in valid_targets]

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

    test_dataset = ImageDataset(
        image_paths=test_image_paths,
        targets=[0] * len(test_image_paths),
        augmentations=test_aug,
    )

    model = FlowerModel(num_classes=len(CLASSES))
    es = EarlyStopping(
        monitor="valid_f1",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
        patience=3,
        mode="max",
        save_weights_only=True,
    )
    model = Tez(model)
    config = TezConfig(
        training_batch_size=TRAIN_BATCH_SIZE,
        validation_batch_size=VALID_BATCH_SIZE,
        epochs=EPOCHS,
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        config=config,
        callbacks=[es],
    )

    model = FlowerModel(num_classes=len(CLASSES))
    model = Tez(model)
    model.load(os.path.join(MODEL_PATH, MODEL_NAME + ".bin"), weights_only=True, device="cuda")
    preds_iter = model.predict(test_dataset)
    final_preds = []
    for preds in preds_iter:
        final_preds.append(preds)
    final_preds = np.vstack(final_preds)
    final_preds = np.argmax(final_preds, axis=1)
    # final_preds = lbl_enc.inverse_transform(final_preds)
    test_image_names = [x.split("/")[-1][:-5] for x in test_image_paths]
    df = pd.DataFrame({"id": test_image_names, "label": final_preds})
    df.to_csv("result.csv", index=False)
