import torch

import numpy as np

from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class RCNNDataset:
    def __init__(
        self, image_paths, bounding_boxes, augmentations=None, torchvision_format=True
    ):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.augmentations = augmentations
        self.torchvision_format = torchvision_format

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        bboxes = self.bounding_boxes[item]
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image, bboxes=bboxes)
            image = augmented["image"]
            bboxes = augmented["bboxes"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        bboxes = np.array(bboxes)

        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        is_crowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": torch.as_tensor(bboxes.tolist(), dtype=torch.float32),
            "area": torch.as_tensor(area.tolist(), dtype=torch.float32),
            "iscrowd": is_crowd,
            "labels": labels,
        }

        if self.torchvision_format:
            return torch.tensor(image, dtype=torch.float), target

        target["image"] = torch.tensor(image, dtype=torch.float)
        return target


class MaskRCNNDataset:
    def __init__(self, image_paths, mask_paths, targets, augmentations=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(img_path + ".png").convert("RGB")
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = rle2mask("1 {}".format(width * height), width, height)
        mask = Image.fromarray(mask.T)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.expand_dims(mask, axis=0)

        pos = np.where(np.array(mask)[0, :, :])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.zeros((1,), dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.ones((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return transforms.ToTensor()(img), target
