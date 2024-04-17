import sys
import json
import argparse

import torch
from torchvision import transforms
from torchgeo.datasets import So2Sat
from tqdm import tqdm


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

    train_dataset = So2Sat(root="./data", version="3_random", split="train",
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

    test_dataset = So2Sat(root="/tmp/data", version="3_random", split="test",
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
    print("loaded datasets", file=sys.stderr)

    return train_dataset, test_dataset


def init_params(m: torch.nn.Module, gain: float = 1.0):
    if m.weight.dim() < 2:
        return
    fan_in = 1.0
    for dim in m.weight.shape[1:]:
        fan_in *= dim
    bound = gain * (3.0 / fan_in)**0.5
    with torch.no_grad():
        m.weight.uniform_(-bound, bound)
        m.bias.uniform_(-bound, bound)


class SimpleRegularisedModel(torch.nn.Module):
    def __init__(self, gain=1.0):
        """initialize the model with 32x32x10 input and 17 outputs"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=32,
                                     kernel_size=3, padding=1)
        init_params(self.conv1, gain=gain*2**0.5)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3, padding=1)
        init_params(self.conv2, gain=gain*2**0.5)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)
        init_params(self.conv3, gain=gain*2**0.5)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*32*32, 128)
        init_params(self.fc1, gain=gain*2**0.5)
        self.fc2 = torch.nn.Linear(128, 64)
        init_params(self.fc2, gain=gain*2**0.5)
        self.fc3 = torch.nn.Linear(64, 17)
        init_params(self.fc3, gain=gain*2**0.5)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.dropout2d = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout2d(x)
        x = self.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = self.relu(self.conv3(x))
        x = self.dropout2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def param_sample(self, indecies: dict[str, torch.Tensor]):
        return {
            name: tensor.ravel()[indecies[name]]
            for name, tensor in self.state_dict().items()}


class SimpleModel(torch.nn.Module):
    def __init__(self, gain=1.0):
        """initialize the model with 32x32x10 input and 17 outputs"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=32,
                                     kernel_size=3, padding=1)
        init_params(self.conv1, gain=gain*2**0.5)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3, padding=1)
        init_params(self.conv2, gain=gain*2**0.5)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)
        init_params(self.conv3, gain=gain*2**0.5)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*32*32, 128)
        init_params(self.fc1, gain=gain*2**0.5)
        self.fc2 = torch.nn.Linear(128, 64)
        init_params(self.fc2, gain=gain*2**0.5)
        self.fc3 = torch.nn.Linear(64, 17)
        init_params(self.fc3, gain=gain*2**0.5)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def param_sample(self, indecies: dict[str, torch.Tensor]):
        return {
            name: tensor.ravel()[indecies[name]]
            for name, tensor in self.state_dict().items()}

class WiderModel(torch.nn.Module):
    def __init__(self):
        """initialize the model with 32x32x10 input and 17 outputs"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=64,
                                     kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128,
                                     kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128,
                                     kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(128*32*32, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 17)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NarrowerModel(torch.nn.Module):
    def __init__(self):
        """initialize the model with 32x32x10 input and 17 outputs"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=10,
                                     kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=10,
                                     kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=10,
                                     kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(10*32*32, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 17)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeeperModel(torch.nn.Module):
    def __init__(self):
        """initialize the model with 32x32x10 input and 17 outputs"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=32,
                                     kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*32*32, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 17)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@torch.no_grad()
def test_model(model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               criterion, device: torch.device):
    model.eval()
    loss = 0
    count = 0
    for data, targets in test_loader:
        output = model(data)
        loss += criterion(output, targets).sum().item()
        count += targets.size(0)
    return loss/count


def train_model(model: torch.nn.Module,
                data: torch.tensor, targets: torch.tensor,
                criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets).sum()
    loss.backward()
    optimizer.step()
    return loss.item()/targets.size(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--model", type=str, default="simple",
                        choices=["simple", "simple_regularised",
                                 "wider", "deeper", "narrower"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}", file=sys.stderr)

    print("loading data", file=sys.stderr)
    train, test = load_so2sat(device=device,
                              train_samples=args.train_samples,
                              test_samples=args.test_samples)

    train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(
            test, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: x)

    print("initializing model", file=sys.stderr)

    model = SimpleModel().to(device)
    match args.model:
        case "simple":
            model = SimpleModel().to(device)
        case "simple_regularised":
            model = SimpleRegularisedModel().to(device)
        case "wider":
            model = WiderModel().to(device)
        case "deeper":
            model = DeeperModel().to(device)
        case "narrower":
            model = NarrowerModel().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("training loop", file=sys.stderr)
    training_batches = 0
    test_loss = test_model(model, test_loader, criterion,
                           device=device)
    print(json.dumps({
        "batch": training_batches,
        "test_loss": test_loss,
    }))
    for epoch in range(args.num_epochs):
        for data, targets in train_loader:
            train_loss = train_model(model, data, targets,
                                     criterion, optimizer)
            training_batches += 1
            if training_batches % args.report_every == 0:
                test_loss = test_model(model, test_loader, criterion,
                                       device=device)
                print(f"Batch {training_batches}, "
                      f"Epoch {epoch+1}/{args.num_epochs}, "
                      f"Training Loss: {train_loss}, "
                      f"Test Loss: {test_loss}", file=sys.stderr)
                print(json.dumps({
                    "batch": training_batches,
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }))
