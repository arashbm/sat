import sys

from typing import Union, Optional
import copy

import torch


DatasetLike = Union[torch.utils.data.Dataset,
                    torch.utils.data.Subset]


def cycle_iterable(iterable):
    while True:
        yield from iterable


class Node:
    def __init__(self, model: torch.nn.Module,
                 train_dataset: DatasetLike,
                 validation_dataset: DatasetLike,
                 batch_size: int = 32):
        self.model = model
        self.data_size = len(train_dataset)
        self.training_dataset = torch.utils.data.dataloader.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda x: x)
        self.validation_dataset = torch.utils.data.dataloader.DataLoader(
                validation_dataset, batch_size=batch_size,
                collate_fn=lambda x: x)

        self.training_iterator = cycle_iterable(self.training_dataset)

    @torch.no_grad()
    def aggregate_neighbours_simple_mean(
            self, neighbours: list["Node"], trusts: list[float],
            param_sample_indecies: Optional[dict[str, torch.Tensor]] = None):
        total_neighbourhood_data = sum(
                neighbour.data_size for neighbour in neighbours)
        total_neighbourhood_data += self.data_size

        # ratio of neighbours' data to total neighbourhood data (inc. self)
        alphas = [neighbour.data_size/total_neighbourhood_data
                  for neighbour in neighbours]

        original_params = copy.deepcopy(self.model.state_dict())

        if len(neighbours) == 0:
            param_changes = {}
            if param_sample_indecies is not None:
                param_changes = {
                    key: 0.0 for key, idxs
                    in param_sample_indecies.items()}
            return original_params, param_changes

        self_trust = 1.
        self_alpha = self.data_size/total_neighbourhood_data
        avg_params = copy.deepcopy(self.model.state_dict())
        for key in avg_params.keys():
            avg_params[key] = torch.zeros_like(avg_params[key],
                                               dtype=torch.float32)

        for key in avg_params.keys():
            for i, neighbour in enumerate(neighbours):
                neighbour_params = neighbour.model.state_dict()
                avg_params[key] += \
                    alphas[i]*trusts[i]*neighbour_params[key]

            avg_params[key] += \
                self_alpha*self_trust*original_params[key]

        param_changes = {}
        if param_sample_indecies is not None:
            param_changes = {
                    key: (avg_params[key].ravel()[idxs] -
                          original_params[key].ravel()[idxs])
                    for key, idxs in param_sample_indecies.items()}

        return avg_params, param_changes

    @torch.no_grad()
    def aggregate_neighbours_decdiff(
            self, neighbours: list["Node"], trusts: list[float],
            param_sample_indecies: Optional[dict[str, torch.Tensor]] = None):
        total_neighbour_data = sum(
                neighbour.data_size for neighbour in neighbours)

        # ratio of neighbours' data to total neighbourhood data (w/o self)
        alphas = [neighbour.data_size/total_neighbour_data
                  for neighbour in neighbours]

        current_params = copy.deepcopy(self.model.state_dict())
        original_params = copy.deepcopy(self.model.state_dict())

        if len(neighbours) == 0:
            return current_params

        avg_params = copy.deepcopy(self.model.state_dict())
        for key in avg_params.keys():
            avg_params[key] = torch.zeros_like(avg_params[key])

        for key in avg_params.keys():
            for i, neighbour in enumerate(neighbours):
                neighbour_params = neighbour.model.state_dict()
                avg_params[key] += \
                    alphas[i]*trusts[i]*neighbour_params[key]

        for key in avg_params.keys():
            dist = current_params[key] - avg_params[key]
            lp_dist = torch.norm(dist, p=2) + 1
            current_params[key] -= dist/lp_dist

        param_changes = {}
        if param_sample_indecies is not None:
            param_changes = {
                    key: (current_params[key].ravel()[idxs] -
                          original_params[key].ravel()[idxs])
                    for key, idxs in param_sample_indecies.items()}

        return current_params, param_changes

    def load_params(self, params: dict):
        self.model.load_state_dict(params)

    def train_simple(
            self, batches: int, optimizer: torch.optim.Optimizer,
            device: torch.device, early_stopping=True,
            param_sample_indecies: Optional[dict[str, torch.Tensor]] = None,
            clamp_loss=True):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        original_params = copy.deepcopy(self.model.state_dict())

        for _, (data, target) in zip(range(batches), self.training_iterator):
            optimizer.zero_grad()
            output = self.model(data)

            loss = torch.mean(criterion(output, target), dim=0)
            if clamp_loss:
                loss = torch.clamp(loss, min=0, max=10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        param_changes = {}
        with torch.no_grad():
            if param_sample_indecies is not None:
                current_params = self.model.state_dict()
                param_changes = {
                    key: (current_params[key].ravel()[idxs] -
                          original_params[key].ravel()[idxs])
                    for key, idxs in param_sample_indecies.items()}
        return param_changes

    @torch.no_grad()
    def validate(self, device: torch.device):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total = 0
        corrects = 0
        losses = 0
        for data, target in self.validation_dataset:
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            loss = criterion(output, target)
            losses += loss.item()
            preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
            corrects += torch.count_nonzero(preds == target).item()
            total += len(data)
        accuracy = corrects/total
        loss = losses/total
        return loss, accuracy

    @torch.no_grad()
    def test(self, test_dataset: DatasetLike, device: torch.device):
        self.model.eval()
        test_dataset = torch.utils.data.dataloader.DataLoader(
                test_dataset, batch_size=64,
                collate_fn=lambda x: x)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total = 0
        corrects = 0
        losses = 0
        for data, target in test_dataset:
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            losses += criterion(output, target).item()
            preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
            corrects += torch.count_nonzero(preds == target).item()
            total += len(data)
        accuracy = corrects/total
        loss = losses/total
        return loss, accuracy


@torch.no_grad()
def stds_across_nodes(
        nodes: list[Node], indecies: dict[str, torch.Tensor]):
    if len(nodes) == 0:
        raise ValueError("at least one model is required")
    states = {name: torch.zeros(len(nodes), indecies[name].numel())
              for name, tensor in nodes[0].model.named_parameters()}

    for i, node in enumerate(nodes):
        for name, tensor in node.model.named_parameters():
            states[name][i] = tensor.ravel()[indecies[name]]

    return {name: tensor.std(dim=0)
            for name, tensor in states.items()}


@torch.no_grad()
def stds_across_params(nodes: list[Node]):
    if len(nodes) == 0:
        raise ValueError("at least one model is required")

    stds = {}
    for i, node in enumerate(nodes):
        stds[i] = {name: tensor.std().item()
                   for name, tensor in node.model.named_parameters()}

    return stds
