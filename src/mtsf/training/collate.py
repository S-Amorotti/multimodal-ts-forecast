import torch


def collate_optional_image(batch):
    """
    Handles batches where img may be None (numeric-only baseline).
    """
    xs, ys, imgs = zip(*batch)

    x = torch.stack(xs)
    y = torch.stack(ys)

    if imgs[0] is None:
        return x, y
    else:
        img = torch.stack(imgs)
        return x, y, img
