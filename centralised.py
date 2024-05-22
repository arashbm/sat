import sys
import json
import argparse

import torch
from tqdm import tqdm

from datasets import load_so2sat
from models import SimpleSo2Sat

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

    model = SimpleSo2Sat().to(device)
    match args.model:
        case "simple":
            model = SimpleSo2Sat().to(device)
        # case "simple_regularised":
        #     model = SimpleRegularisedModel().to(device)
        # case "wider":
        #     model = WiderModel().to(device)
        # case "deeper":
        #     model = DeeperModel().to(device)
        # case "narrower":
        #     model = NarrowerModel().to(device)
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
