import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        if len(batch) == 3:
            x, y, img = batch
            x, y, img = x.to(device), y.to(device), img.to(device)
            pred = model(x, img)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)

        loss = torch.mean((pred - y) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae, mse = 0.0, 0.0

    for batch in loader:
        if len(batch) == 3:
            x, y, img = batch
            x, y, img = x.to(device), y.to(device), img.to(device)
            pred = model(x, img)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)

        mae += torch.mean(torch.abs(pred - y)).item() * x.size(0)
        mse += torch.mean((pred - y) ** 2).item() * x.size(0)

    mae /= len(loader.dataset)
    rmse = (mse / len(loader.dataset)) ** 0.5

    return mae, rmse
