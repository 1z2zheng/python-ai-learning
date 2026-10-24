from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
CHANNELS = {
    "R": (0,),
    "G": (1,),
    "B": (2,),
    "RG": (0, 1),
    "RB": (0, 2),
    "GB": (1, 2),
    "RGB": (0, 1, 2),
}


def find_image(directory: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到图像：{directory / stem}")


def load_split(data_root: Path, split: str) -> list[str]:
    split_file = data_root / "splits" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"缺少数据划分文件：{split_file}")
    stems = [line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not stems:
        raise ValueError(f"数据划分为空：{split_file}")
    return stems


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int,
        num_classes: int,
        channel_mode: str = "RGB",
        augment: bool = False,
        augmentation_config: dict[str, object] | None = None,
    ) -> None:
        self.root = Path(data_root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.stems = load_split(self.root, split)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.channel_mode = channel_mode.upper()
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        if self.channel_mode not in CHANNELS:
            raise ValueError(f"不支持的channel_mode：{channel_mode}")

    @property
    def in_channels(self) -> int:
        return len(CHANNELS[self.channel_mode])

    def __len__(self) -> int:
        return len(self.stems)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        config = self.augmentation_config
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        image = image.astype(np.float32, copy=True)
        if random.random() < float(config.get("brightness_probability", 0.7)):
            image *= random.uniform(
                float(config.get("brightness_min", 0.80)),
                float(config.get("brightness_max", 1.20)),
            )
        if random.random() < float(config.get("contrast_probability", 0.7)):
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = (image - mean) * random.uniform(
                float(config.get("contrast_min", 0.80)),
                float(config.get("contrast_max", 1.20)),
            ) + mean
        if random.random() < float(config.get("gamma_probability", 0.0)):
            gamma = random.uniform(
                float(config.get("gamma_min", 0.75)),
                float(config.get("gamma_max", 1.45)),
            )
            image = 255.0 * np.power(np.clip(image, 0, 255) / 255.0, gamma)
        if random.random() < float(config.get("channel_gain_probability", 0.5)):
            channel_gain = np.asarray(
                [
                    random.uniform(
                        float(config.get("channel_gain_min", 0.90)),
                        float(config.get("channel_gain_max", 1.10)),
                    )
                    for _ in range(3)
                ],
                dtype=np.float32,
            )
            image *= channel_gain.reshape(1, 1, 3)
        # Random erasing simulates leaf/stem occlusion.  Only the image is erased;
        # the original target teaches the network to recover the complete head region.
        if random.random() < float(config.get("occlusion_probability", 0.0)):
            height, width = image.shape[:2]
            count = random.randint(1, int(config.get("occlusion_max_count", 3)))
            min_ratio = float(config.get("occlusion_min_ratio", 0.03))
            max_ratio = float(config.get("occlusion_max_ratio", 0.12))
            for _ in range(count):
                erase_h = max(1, int(height * random.uniform(min_ratio, max_ratio)))
                erase_w = max(1, int(width * random.uniform(min_ratio, max_ratio)))
                top = random.randint(0, max(height - erase_h, 0))
                left = random.randint(0, max(width - erase_w, 0))
                fill = image.mean(axis=(0, 1), keepdims=True)
                image[top : top + erase_h, left : left + erase_w] = fill
        return np.clip(image, 0, 255), mask.copy()

    def __getitem__(self, index: int) -> dict[str, object]:
        stem = self.stems[index]
        image_path = find_image(self.images_dir, stem)
        mask_path = find_image(self.masks_dir, stem)

        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        image_array = np.asarray(image, dtype=np.float32)
        mask_array = np.asarray(mask, dtype=np.int64)
        if mask_array.min() < 0 or mask_array.max() >= self.num_classes:
            raise ValueError(
                f"{mask_path} 的类别值范围为 {mask_array.min()}~{mask_array.max()}，"
                f"但num_classes={self.num_classes}"
            )
        if self.augment:
            image_array, mask_array = self._augment(image_array, mask_array)

        image_array = image_array[:, :, CHANNELS[self.channel_mode]] / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask_array).long()
        return {"image": image_tensor, "mask": mask_tensor, "stem": stem}

