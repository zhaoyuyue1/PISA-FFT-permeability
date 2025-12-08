from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


class PermeabilityDataset(Dataset):
    """
    Dataset for permeability prediction on porous media images.

    Each CSV row is expected to contain:
      - image identifier in the first column (file name without extension)
      - permeability value in the second column

    For each image we:
      1. Load the corresponding RGB PNG from `image_dir`.
      2. Apply Albumentations preprocessing and normalization.
      3. Compute patch-level physics features (porosity and area ratio)
         for each color channel using Otsu thresholding and connected components.
      4. Return the preprocessed image tensor, patch features and log(1 + k) label.
    """

    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        patch_size: int = 56,
        img_size: int = 224,
        transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.patch_size = patch_size
        self.img_size = img_size

        # Default transform: resize + normalize + to tensor
        self.transform = transform or A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_id = str(self.data.iloc[idx, 0])
        img_path = os.path.join(self.image_dir, img_id + ".png")

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image_rgb)
        tensor_chw = transformed["image"]  # [C, H, W]
        # For patch feature extraction we temporarily convert back to HWC
        image = tensor_chw.numpy().transpose(1, 2, 0)

        ps = self.patch_size
        h, w, c = image.shape
        if h % ps != 0 or w % ps != 0:
            raise ValueError(
                f"Image size ({h}, {w}) must be divisible by patch_size={ps}"
            )

        num_patches_h, num_patches_w = h // ps, w // ps
        patch_feats = []

        # For each channel, compute [porosity, max-connected-area ratio]
        for ch in range(c):
            feats = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = image[
                        i * ps : (i + 1) * ps,
                        j * ps : (j + 1) * ps,
                        ch,
                    ]
                    thresh = threshold_otsu(patch)
                    bw = (patch > thresh).astype(np.uint8)

                    poro = float(np.mean(bw))

                    lbl = label(bw, connectivity=2)
                    props = regionprops(lbl)
                    if props:
                        max_area = max(prop.area for prop in props)
                        area_ratio = float(max_area) / float(ps * ps)
                    else:
                        area_ratio = 0.0
                    feats.append([poro, area_ratio])
            feats = np.array(feats, dtype=np.float32)
            patch_feats.append(feats)

        # Concatenate features for all channels: [N_patches, 2 * channels] = [N, 6]
        patch_feats_arr = np.concatenate(patch_feats, axis=1)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        patch_feats_tensor = torch.from_numpy(patch_feats_arr).float()

        permeability = float(self.data.iloc[idx, 1])
        log_label = np.log1p(permeability)
        label_tensor = torch.tensor(log_label, dtype=torch.float32)

        return image_tensor, patch_feats_tensor, label_tensor


def build_dataloaders(
    image_dir: str,
    csv_file: str,
    patch_size: int,
    img_size: int,
    batch_size: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test dataloaders with a random split.

    If `seed` is None, the default PyTorch RNG state is used for the split.
    """
    dataset = PermeabilityDataset(
        image_dir=image_dir,
        csv_file=csv_file,
        patch_size=patch_size,
        img_size=img_size,
    )

    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None  # non-deterministic split

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
