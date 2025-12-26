import numpy as np


class ShuffledImageTransform:
    """
    Wraps an image transform and shuffles outputs across calls.
    Used for ablation.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.buffer = []

    def __call__(self, x):
        img = self.base_transform(x)
        self.buffer.append(img)

        if len(self.buffer) < 2:
            return img

        idx = np.random.randint(len(self.buffer))
        return self.buffer[idx]
