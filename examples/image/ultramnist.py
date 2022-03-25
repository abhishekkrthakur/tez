import argparse
import os

import albumentations
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn import metrics, model_selection
from tqdm import tqdm

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.utils import seed_everything
from sklearn import preprocessing


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tf_efficientnetv2_b1", required=False)
    parser.add_argument("--learning_rate", type=float, default=1e-2, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--epochs", type=int, default=150, required=False)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--image_size", type=int, default=256, required=False)
    return parser.parse_args()


def img_resize(path, args, is_train):
    image = cv2.imread(path)
    image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
    base = os.path.basename(path)
    if is_train:
        cv2.imwrite(os.path.join(args.output, "train", f"{base}"), image)
    else:
        cv2.imwrite(os.path.join(args.output, "test", f"{base}"), image)


class UltraMNISTDataset:
    def __init__(self, image_paths, targets, augmentations):
        self.image_paths = image_paths
        self.augmentations = augmentations
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        targets = self.targets[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


class UltraMNISTModel(nn.Module):
    def __init__(self, model_name, num_classes, learning_rate, n_train_steps):
        super().__init__()

        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        self.model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
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
    os.makedirs(os.path.join(args.output, f"train"), exist_ok=True)
    os.makedirs(os.path.join(args.output, f"test"), exist_ok=True)

    df = pd.read_csv(os.path.join(args.input, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.input, "sample_submission.csv"))
    test_df.loc[:, "digit_sum"] = 0  # fake targets for test data
    test_ids = test_df["id"].values

    train_img_paths = [os.path.join(args.input, "train", f"{i}.jpeg") for i in df["id"].values]
    test_img_paths = [os.path.join(args.input, "test", f"{i}.jpeg") for i in test_df["id"].values]

    # resize images
    Parallel(n_jobs=16)(delayed(img_resize)(path, args, is_train=True) for path in tqdm(train_img_paths))
    Parallel(n_jobs=16)(delayed(img_resize)(path, args, is_train=False) for path in tqdm(test_img_paths))

    train_aug = albumentations.Compose(
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

    lbl_enc = preprocessing.LabelEncoder()
    df["digit_sum"] = lbl_enc.fit_transform(df["digit_sum"].values)

    train_df, valid_df = model_selection.train_test_split(
        df,
        test_size=0.5,
        random_state=42,
        stratify=df["digit_sum"].values,
    )

    train_ids = train_df["id"].values
    valid_ids = valid_df["id"].values

    # use resized images here
    train_img_paths = [os.path.join(args.output, "train", f"{i}.jpeg") for i in train_ids]
    valid_img_paths = [os.path.join(args.output, "train", f"{i}.jpeg") for i in valid_ids]
    test_img_paths = [os.path.join(args.output, "test", f"{i}.jpeg") for i in test_ids]

    train_dataset = UltraMNISTDataset(
        image_paths=train_img_paths,
        targets=train_df["digit_sum"].values,
        augmentations=train_aug,
    )

    valid_dataset = UltraMNISTDataset(
        image_paths=valid_img_paths,
        targets=valid_df["digit_sum"].values,
        augmentations=valid_aug,
    )

    test_dataset = UltraMNISTDataset(
        image_paths=test_img_paths,
        targets=test_df["digit_sum"].values,
        augmentations=valid_aug,
    )

    n_train_steps = int(len(train_img_paths) / args.batch_size / args.accumulation_steps * args.epochs)
    model = UltraMNISTModel(
        model_name=args.model_name,
        num_classes=df.digit_sum.nunique(),
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

    model.load(os.path.join(args.output, "model.bin"), weights_only=True)

    preds_iter = model.predict(test_dataset)
    final_preds = []
    for preds in preds_iter:
        print(preds)
        final_preds.append(preds)
    final_preds = np.vstack(final_preds)
    final_preds = np.argmax(final_preds, axis=1)
    final_preds = lbl_enc.inverse_transform(final_preds)

    df = pd.DataFrame({"id": test_ids, "digit_sum": final_preds})
    df.to_csv(os.path.join(args.output, "submission.csv"), index=False)
