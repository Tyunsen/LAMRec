import math
import random
from typing import List
import torch


def str2bool(x):
    if x == 'True':
        return True
    elif x == 'False':
        return False


def generate_random_seed():
    return random.randint(0, 2 ** 32 - 1) % 10000000000


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def generate_3dTo2d_sample(input: List[List[List[str]]]) -> List[List[str]]:
    res = []
    for i in range(len(input)):
        for j in range(len(input[i])):
            res.append(input[i][j])
    return res


def generate_aug_sample(input: List[List[str]]) -> List[List[str]]:
    res = []
    for i in range(len(input)):
        num_mask = math.floor(len(input[i]) * 0.3)  # 0.3 is a params
        mask_index = random.sample(range(len(input[i])), k=num_mask)
        visit_aug = [key for index, key in enumerate(input[i]) if index not in mask_index]
        res.append(visit_aug)
    return res


def add_drug_pad(input: List[List[str]], pad: int) -> List[List[str]]:
    max_len = 0
    for i in range(len(input)):
        max_len = max(len(input[i]), max_len)
    return [drug + [pad] * (max_len - len(drug)) for drug in input]


def mul2idx(tensor, pad=0):
    res = []
    for sample in tensor:
        idxs = torch.nonzero((sample > 0.5)).squeeze()
        idxs = idxs[idxs > 1]
        idx_sorted = torch.argsort(sample[idxs], descending=True)
        if len(idxs) > 50:
            idx_topk = idxs[idx_sorted][:50]
        else:
            idx_topk = idxs[idx_sorted]
        res.append([int(idx) for idx in idx_topk])
    max_len = 0
    for item in res:
        max_len = max(max_len, len(item))
    return [item + [pad] * (max_len - len(item)) for item in res]


def batch_to_multihot(label: List[List[int]], num_labels: int) -> torch.tensor:
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot


def batch_to_multihot_x(label: List[List[int]], num_labels: int) -> torch.tensor:
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
        multihot[i, 0:2] = 0
    return multihot
