import argparse
import os

import albumentations
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from cv2 import dft
from sklearn import metrics, model_selection
from tqdm.auto import tqdm

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
from tez.utils import seed_everything


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tf_efficientnet_b3_ns", required=False)
    parser.add_argument("--learning_rate", type=float, default=1e-3, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--epochs", type=int, default=50, required=False)
    parser.add_argument("--output", type=str, default="~/data/", required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--image_size", type=int, default=512, required=False)
    return parser.parse_args()


class SorghumModel(nn.Module):
    def __init__(self, model_name, num_classes, learning_rate, n_train_steps):
        super().__init__()

        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=3,
            num_classes=num_classes,
        )

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        targets = targets.cpu().detach().numpy()
        acc = metrics.accuracy_score(targets, outputs)
        acc = torch.tensor(acc, device=device)
        return {"accuracy": acc}

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            patience=2,
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        return opt, sch

    def forward(self, image, targets=None):
        x = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}


if __name__ == "__main__":

    args = parse_args()
    seed_everything(42)
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(os.path.join(args.input, "train_cultivar_mapping.csv"))
    test_df = pd.read_csv(os.path.join(args.input, "sample_submission.csv"))

    unique_labels = df["cultivar"].unique()
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    rev_label_mapping = {i: label for label, i in label_mapping.items()}

    df.loc[:, "cultivar"] = df["cultivar"].map(label_mapping)
    test_df.loc[:, "cultivar"] = test_df["cultivar"].map(label_mapping)

    train_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(
                height=args.image_size,
                width=args.image_size,
                p=1,
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HueSaturationValue(p=0.5),
            albumentations.OneOf(
                [
                    albumentations.RandomBrightnessContrast(p=0.5),
                    albumentations.RandomGamma(p=0.5),
                ],
                p=0.5,
            ),
            albumentations.OneOf(
                [
                    albumentations.Blur(p=0.1),
                    albumentations.GaussianBlur(p=0.1),
                    albumentations.MotionBlur(p=0.1),
                ],
                p=0.1,
            ),
            albumentations.OneOf(
                [
                    albumentations.GaussNoise(p=0.1),
                    albumentations.ISONoise(p=0.1),
                    albumentations.GridDropout(ratio=0.5, p=0.2),
                    albumentations.CoarseDropout(
                        max_holes=16,
                        min_holes=8,
                        max_height=16,
                        max_width=16,
                        min_height=8,
                        min_width=8,
                        p=0.2,
                    ),
                ],
                p=0.2,
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
            albumentations.Resize(
                height=args.image_size,
                width=args.image_size,
                p=1.0,
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

    train_df, valid_df = model_selection.train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["cultivar"].values,
    )

    train_image_paths = [os.path.join(args.input, "train", x) for x in train_df["image"].values]
    valid_image_paths = [os.path.join(args.input, "train", x) for x in valid_df["image"].values]
    test_image_paths = [
        os.path.join(args.input, "test", x.replace(".png", ".jpeg")) for x in test_df["filename"].values
    ]

    train_targets = train_df["cultivar"].values
    valid_targets = valid_df["cultivar"].values
    test_targets = test_df["cultivar"].values

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=train_aug,
    )
    valid_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=valid_aug,
    )
    test_dataset = ImageDataset(
        image_paths=test_image_paths,
        targets=test_targets,
        augmentations=valid_aug,
    )

    n_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    model = SorghumModel(
        model_name=args.model_name,
        num_classes=len(label_mapping),
        learning_rate=args.learning_rate,
        n_train_steps=n_train_steps,
    )

    model = Tez(model)
    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=2 * args.batch_size,
        test_batch_size=2 * args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_accuracy",
        fp16=True,
    )

    es = EarlyStopping(
        monitor="valid_accuracy",
        model_path=os.path.join(args.output, "model.bin"),
        patience=10,
        mode="max",
        save_weights_only=True,
    )

    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[es],
        config=config,
    )

    model.load(os.path.join(args.output, "model.bin"), weights_only=True, config=config)

    preds_iter = model.predict(test_dataset)
    final_preds = []
    for preds in tqdm(preds_iter, total=(len(test_dataset) / (2 * args.batch_size))):
        final_preds.append(preds)
    final_preds = np.vstack(final_preds)
    final_preds = np.argmax(final_preds, axis=1)

    df = pd.DataFrame(
        {
            "filename": test_df["filename"].values,
            "cultivar": final_preds,
        }
    )
    df.loc[:, "cultivar"] = df["cultivar"].map(rev_label_mapping)
    df.to_csv(os.path.join(args.output, "submission.csv"), index=False)
