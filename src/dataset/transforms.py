import torch
import numpy as np
import random
import scipy.ndimage as ndi

class Compose3D:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, label: np.ndarray = None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class RandomFlip3D:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: np.ndarray, label: np.ndarray = None):
        # Image shape: (C, D, H, W) or (D, H, W)
        # Flip along axial or coronal axis (avoid sagittal flip if Left/Right classes are distinct, or swap classes 1 and 2 if flipped horizontally!)
        # 1. Flip along D (z-axis, superior-inferior - rarely upside down, but minor chance)
        if random.random() < 0.2:
            image = np.flip(image, axis=-3).copy()
            if label is not None:
                label = np.flip(label, axis=-3).copy()
                
        # 2. Flip along H (y-axis, anterior-posterior)
        if random.random() < self.prob:
            image = np.flip(image, axis=-2).copy()
            if label is not None:
                label = np.flip(label, axis=-2).copy()
                
        # 3. Flip along W (x-axis, sagittal Left-Right flip)
        # IMPORTANT: When flipping left/right, Class 1 (Right IAN) and Class 2 (Left IAN) MUST be swapped!
        if random.random() < self.prob:
            image = np.flip(image, axis=-1).copy()
            if label is not None:
                label = np.flip(label, axis=-1).copy()
                mask_r = (label == 1)
                mask_l = (label == 2)
                label[mask_r] = 2
                label[mask_l] = 1
                
        return image, label

class RandomGaussianNoise3D:
    def __init__(self, prob: float = 0.3, mean: float = 0.0, std: float = 0.05):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, label: np.ndarray = None):
        if random.random() < self.prob:
            noise = np.random.normal(self.mean, self.std, size=image.shape).astype(image.dtype)
            image = image + noise
        return image, label

class RandomIntensityScaling3D:
    def __init__(self, prob: float = 0.3, scale_range: tuple = (0.85, 1.15)):
        self.prob = prob
        self.scale_range = scale_range

    def __call__(self, image: np.ndarray, label: np.ndarray = None):
        if random.random() < self.prob:
            factor = random.uniform(*self.scale_range)
            image = image * factor
        return image, label

class RandomGamma3D:
    def __init__(self, prob: float = 0.3, gamma_range: tuple = (0.7, 1.5)):
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, image: np.ndarray, label: np.ndarray = None):
        if random.random() < self.prob:
            gamma = random.uniform(*self.gamma_range)
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min + 1e-8)
                transformed = np.power(normalized, gamma)
                image = transformed * (img_max - img_min) + img_min
        return image, label

class RandomCutout3D:
    def __init__(self, prob: float = 0.25, max_box_ratio: float = 0.2):
        self.prob = prob
        self.max_box_ratio = max_box_ratio

    def __call__(self, image: np.ndarray, label: np.ndarray = None):
        if random.random() < self.prob:
            d, h, w = image.shape[-3:]
            bd, bh, bw = int(d * self.max_box_ratio), int(h * self.max_box_ratio), int(w * self.max_box_ratio)
            z = random.randint(0, max(1, d - bd))
            y = random.randint(0, max(1, h - bh))
            x = random.randint(0, max(1, w - bw))
            
            if image.ndim == 4:
                image[:, z:z+bd, y:y+bh, x:x+bw] = 0
            else:
                image[z:z+bd, y:y+bh, x:x+bw] = 0
        return image, label

def get_training_transforms() -> Compose3D:
    return Compose3D([
        RandomFlip3D(prob=0.5),
        RandomGaussianNoise3D(prob=0.3, std=0.08),
        RandomIntensityScaling3D(prob=0.35, scale_range=(0.85, 1.15)),
        RandomGamma3D(prob=0.3, gamma_range=(0.75, 1.4)),
        RandomCutout3D(prob=0.2, max_box_ratio=0.15)
    ])

def get_validation_transforms() -> Compose3D:
    return Compose3D([])
