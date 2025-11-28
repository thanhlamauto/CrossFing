# jax_data.py
import random
import os
from typing import List, Tuple, Iterator, Dict, Optional

import cv2
import numpy as np


def read_pair_list(npy_path: str) -> List[str]:
    """
    Giống train.npy/valid.npy của JIPNet: np.load(...).item()['info_lst'].
    """
    data = np.load(npy_path, allow_pickle=True).item()
    return data["info_lst"]


def parse_pair_txt(txt_path: str, data_root: Optional[str] = None) -> Tuple[str, str, float]:
    """
    Đọc 1 file .txt:
    line[1]: path img1
    line[2]: path img2
    line[5]: gt (0/1)       (hoặc fallback line[3])
    
    If data_root is provided, prepend it to relative paths.
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()
    fpath1 = lines[1].strip()
    fpath2 = lines[2].strip()
    if len(lines) > 5:
        gt = float(lines[5].strip().split()[0])
    else:
        gt = float(lines[3].strip().split()[0])
    
    # Prepend data_root if provided and paths are relative
    if data_root is not None:
        if not os.path.isabs(fpath1):
            fpath1 = os.path.join(data_root, fpath1)
        if not os.path.isabs(fpath2):
            fpath2 = os.path.join(data_root, fpath2)
    
    return fpath1, fpath2, gt


def load_img_and_mask(path: str,
                      input_size: int = 320,
                      use_mask: bool = True):
    """
    Đọc ảnh grayscale, resize về input_size, invert 255-x, scale về [0,1].
    Mask: path*_mask.png (Verifinger), nếu không có thì tất cả foreground.
    """
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = img.astype(np.float32)
    img = cv2.resize(img, (input_size, input_size),
                     interpolation=cv2.INTER_LINEAR)

    img = (255.0 - img) / 255.0
    img = img[..., None]  # [H,W,1]

    if use_mask:
        mask_path = path.replace(".png", "_mask.png")
        m = cv2.imread(mask_path, 0)
        if m is None:
            m = np.ones_like(img[..., 0], dtype=np.float32) * 255
        m = cv2.resize(m, (input_size, input_size),
                       interpolation=cv2.INTER_NEAREST)
        m = (m < 128).astype(np.float32)[..., None]  # [H,W,1]
    else:
        m = np.ones_like(img, dtype=np.float32)

    return img, m


def augment(img, mask):
    """
    Augmentation nhẹ: rotate ±10°. Bạn có thể thêm TPS/elastic sau này.
    img, mask: [H,W,1]
    """
    angle = np.random.uniform(-10, 10)
    h, w, _ = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    img2 = cv2.warpAffine(img[..., 0], M, (w, h), borderValue=255)
    m2 = cv2.warpAffine(mask[..., 0], M, (w, h), borderValue=0)

    img2 = img2[..., None]
    m2 = m2[..., None]
    return img2, m2


def pair_generator(info_lst: List[str],
                   batch_size: int,
                   input_size: int = 320,
                   use_augmentation: bool = True,
                   use_mask: bool = True,
                   shuffle: bool = True,
                   data_root: Optional[str] = None) -> Iterator[Dict[str, np.ndarray]]:
    """
    Yield batch dict: {
      'img1': [B,H,W,1],
      'img2': [B,H,W,1],
      'mask1': [B,H,W,1],
      'mask2': [B,H,W,1],
      'target': [B,1]
    }
    
    Args:
        data_root: Optional root directory to prepend to image paths
    """
    idxs = list(range(len(info_lst)))
    while True:
        if shuffle:
            random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i + batch_size]
            imgs1, imgs2, masks1, masks2, targets = [], [], [], [], []
            for idx in batch_idx:
                fpath1, fpath2, gt = parse_pair_txt(info_lst[idx], data_root=data_root)
                img1, m1 = load_img_and_mask(fpath1, input_size, use_mask)
                img2, m2 = load_img_and_mask(fpath2, input_size, use_mask)

                if use_augmentation:
                    if random.random() > 0.5:
                        img1, m1 = augment(img1, m1)
                    if random.random() > 0.5:
                        img2, m2 = augment(img2, m2)

                imgs1.append(img1)
                imgs2.append(img2)
                masks1.append(m1)
                masks2.append(m2)
                targets.append([gt])

            batch = {
                "img1": np.stack(imgs1, axis=0).astype(np.float32),
                "img2": np.stack(imgs2, axis=0).astype(np.float32),
                "mask1": np.stack(masks1, axis=0).astype(np.float32),
                "mask2": np.stack(masks2, axis=0).astype(np.float32),
                "target": np.stack(targets, axis=0).astype(np.float32),
            }
            yield batch
