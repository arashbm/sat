from typing import Optional
import sys

import numpy as np
import torch


def zipf_probs(alpha: float, users: int, support: list[int],
               random_state: np.random.Generator):
    samples = random_state.zipf(alpha, users)
    while sum(samples) > min(support):
        samples = random_state.zipf(alpha, users)

    return samples/sum(samples)


def zipf_sampler(alpha: float, users: int, dataset: torch.utils.data.Dataset,
                 validation_split: float,
                 random_state: np.random.Generator):
    targets = torch.tensor([t for _, t in dataset])
    label_indeces = torch.unique(targets)
    classes = len(label_indeces)
    support = [torch.count_nonzero(targets == c).item()
               for c in label_indeces]

    probs = zipf_probs(alpha, users, support, random_state)

    partitions = [[] for u in range(users)]
    for c in range(classes):
        user_probs = np.roll(probs, c)
        class_ids = torch.where(targets == c)[0].numpy()
        random_state.shuffle(class_ids)

        offset = 0
        for user, p in enumerate(user_probs):
            upper_bound = offset + int(p*class_ids.shape[0])
            partitions[user].extend(
                    class_ids[offset:upper_bound])

            offset = upper_bound

    user_splits = []
    for i, ids in enumerate(partitions):
        ids = random_state.permutation(ids)
        val_size = int(validation_split*len(ids))
        training = torch.utils.data.Subset(dataset, ids[val_size:])
        validation = torch.utils.data.Subset(dataset, ids[:val_size])
        user_splits.append((training, validation))

    print("trianing splits:",
          [len(t) for t, _ in user_splits], file=sys.stderr)
    print("validation splits:",
          [len(v) for _, v in user_splits], file=sys.stderr)

    return user_splits


def balanced_iid_sampler(users: int, dataset: torch.utils.data.Dataset,
                         validation_split: float,
                         random_state: np.random.Generator,
                         items_per_user: Optional[int] = None):
    targets = torch.tensor([t for _, t in dataset])
    label_indeces = torch.unique(targets)
    classes = len(label_indeces)

    if items_per_user is None:
        smallest_class = min(
                torch.count_nonzero(targets == c).item()
                for c in range(classes))
        items_per_user = (smallest_class*classes)//users
        print(smallest_class, file=sys.stderr)

    items_per_class = items_per_user//classes
    training_items_per_class = int(items_per_class*(1-validation_split))

    user_training = [[] for i in range(users)]
    user_validation = [[] for i in range(users)]
    for c in range(classes):
        class_ids = torch.where(targets == c)[0].numpy()
        random_state.shuffle(class_ids)
        for u in range(users):
            user_training[u].extend(
                class_ids[
                    u*items_per_class:
                    u*items_per_class+training_items_per_class])
            user_validation[u].extend(
                class_ids[
                    u*items_per_class+training_items_per_class:
                    (u+1)*items_per_class])

    print("training splits:",
          [len(u) for u in user_training], file=sys.stderr)
    print("validation splits:",
          [len(u) for u in user_validation], file=sys.stderr)

    return [(torch.utils.data.Subset(dataset, t),
             torch.utils.data.Subset(dataset, v))
            for t, v in zip(user_training, user_validation)]


def iid_sampler(users: int, dataset: torch.utils.data.Dataset,
                validation_split: float,
                random_state: np.random.Generator,
                items_per_user: Optional[int] = None):
    if items_per_user is None:
        items_per_user = len(dataset)//users
    training_items_per_user = int(items_per_user*(1-validation_split))
    ids = torch.randperm(len(dataset))

    if len(ids) < users*items_per_user:
        raise ValueError("not enough samples")

    datasets = []
    for i in range(users):
        start = i*items_per_user
        middle = start + training_items_per_user
        end = (i+1)*items_per_user
        t = torch.utils.data.Subset(dataset, ids[start:middle].tolist())
        v = torch.utils.data.Subset(dataset, ids[middle:end].tolist())

        datasets.append((t, v))

    print("training splits:",
          [len(t) for t, _ in datasets], file=sys.stderr)
    print("validation splits:",
          [len(v) for _, v in datasets], file=sys.stderr)

    return datasets


def print_partition_counts(partitions):
    train_unique = set()
    valid_unique = set()
    for t, v in partitions:
        train_counts = {}
        valid_counts = {}
        for item, label in t:
            label = label.item()
            train_unique.add(item)
            if label not in train_counts:
                train_counts[label] = 0
            train_counts[label] += 1

        for item, label in v:
            label = label.item()
            valid_unique.add(item)
            if label not in valid_counts:
                valid_counts[label] = 0
            valid_counts[label] += 1
        print("counts:", train_counts, valid_counts, file=sys.stderr)
    print("unique counts:", len(train_unique), len(valid_unique),
          file=sys.stderr)
