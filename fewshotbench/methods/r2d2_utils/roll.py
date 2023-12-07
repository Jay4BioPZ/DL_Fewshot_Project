import torch

def roll(x, shift):
    return torch.cat((x[-shift:], x[:-shift]))