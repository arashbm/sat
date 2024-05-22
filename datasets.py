import sys

import torch
from torchvision import transforms
from tqdm import tqdm
from torchgeo.datasets import So2Sat
from torchvision.datasets import MNIST, CIFAR10


class PairDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    data: torch.Tensor
    targest: torch.Tensor

    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        assert targets.size(0) == data.size(0), "Size mismatch between tensors"
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return (self.data[index], self.targets[index])

    def __getitems__(self, indices: list):
        return (self.data[indices], self.targets[indices])

    def __len__(self):
        return self.data.size(0)


def load_mnist(device: torch.device = torch.device("cpu"),
               train_samples=10000, test_samples=1000):
    print(f"preparing datasets {train_samples} {test_samples}",
          file=sys.stderr)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and Std deviation for MNIST
    ])

    train_dataset = MNIST(root="data", train=True, download=True, transform=normalize)
    print(f"Total training dataset size: {len(train_dataset)}", file=sys.stderr)

    if train_samples > len(train_dataset):
        raise ValueError("Training samples must not be larger than the training dataset")

    train_indices = torch.randperm(len(train_dataset))[:train_samples]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    print("Reading training dataset", file=sys.stderr)

    data, targets = zip(*tqdm(train_dataset))
    train_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))

    test_dataset = MNIST(root="data", train=False, download=True, transform=normalize)
    print(f"Total test dataset size: {len(test_dataset)}", file=sys.stderr)
    if test_samples > len(test_dataset):
        raise ValueError("Test samples must not be larger than the test dataset")

    test_indices = torch.randperm(len(test_dataset))[:test_samples]
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    print("Reading test dataset", file=sys.stderr)
    data, targets = zip(*tqdm(test_dataset))
    test_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))

    print("Loaded MNIST datasets", file=sys.stderr)

    return train_dataset, test_dataset


def load_so2sat(device: torch.device = torch.device("cpu"),
                train_samples=10000, test_samples=1000):
    print(f"preparing datasets {train_samples} {test_samples}",
          file=sys.stderr)

    bands = ('S2_B02', 'S2_B03', 'S2_B04', 'S2_B05', 'S2_B06', 'S2_B07',
             'S2_B08', 'S2_B8A', 'S2_B11', 'S2_B12')
    band_means = torch.tensor([
        0.12375696117681859,
        0.1092774636368323,
        0.1010855203267882,
        0.1142398616114001,
        0.1592656692023089,
        0.18147236008771792,
        0.1745740312291377,
        0.19501607349635292,
        0.15428468872076637,
        0.10905050699570007,
    ])

    band_stds = torch.tensor([
        0.03958795985905458,
        0.047778262752410296,
        0.06636616706371974,
        0.06358874912497474,
        0.07744387147984592,
        0.09101635085921553,
        0.09218466562387101,
        0.10164581233948201,
        0.09991773043519253,
        0.08780632509122865,
    ])

    def normalise(item):
        return {"label": item["label"],
                "image": transforms.functional.normalize(
                    item["image"], mean=band_means, std=band_stds)}

    train_dataset = So2Sat(root="data", version="3_random", split="train",
                           bands=bands, transforms=normalise)
    print(len(train_dataset), file=sys.stderr)
    if train_samples > len(train_dataset):
        raise ValueError("training samples must not be larger than the "
                         "training dataset")
    train_indices = torch.randperm(len(train_dataset))[:train_samples]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    print("reading training dataset", file=sys.stderr)
    print(f"dataset size {len(train_dataset)}", file=sys.stderr)
    data, targets = zip(*((d["image"], d["label"])
                          for d in tqdm(train_dataset)))
    train_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))

    test_dataset = So2Sat(root="data", version="3_random", split="test",
                          bands=bands, transforms=normalise)
    print(len(test_dataset), file=sys.stderr)
    if test_samples > len(test_dataset):
        raise ValueError("test samples must not be larger than the "
                         "test dataset")
    test_indices = torch.randperm(len(test_dataset))[:test_samples]
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    print("reading test dataset", file=sys.stderr)
    data, targets = zip(*((d["image"], d["label"])
                          for d in tqdm(test_dataset)))
    test_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))
    print("loaded So2Sat datasets", file=sys.stderr)

    return train_dataset, test_dataset


def load_cifar10(device: torch.device = torch.device("cpu"),
                  train_samples=10000, test_samples=1000):
    print(f"preparing datasets {train_samples} {test_samples}",
          file=sys.stderr)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CIFAR10(root="data", train=True, download=True, transform=normalize)
    print(f"Total training dataset size: {len(train_dataset)}", file=sys.stderr)

    if train_samples > len(train_dataset):
        raise ValueError("Training samples must not be larger than the training dataset")

    train_indices = torch.randperm(len(train_dataset))[:train_samples]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    print("Reading training dataset", file=sys.stderr)

    data, targets = zip(*tqdm(train_dataset))
    train_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))

    test_dataset = CIFAR10(root="data", train=False, download=True, transform=normalize)
    print(f"Total test dataset size: {len(test_dataset)}", file=sys.stderr)
    if test_samples > len(test_dataset):
        raise ValueError("Test samples must not be larger than the test dataset")

    test_indices = torch.randperm(len(test_dataset))[:test_samples]
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    print("Reading test dataset", file=sys.stderr)
    data, targets = zip(*tqdm(test_dataset))
    test_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))

    print("Loaded CIFAR10 datasets", file=sys.stderr)

    return train_dataset, test_dataset
