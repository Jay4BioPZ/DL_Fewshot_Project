import torch
from torch.autograd import Variable

def make_float_label(n_way, n_samples):
    label = torch.FloatTensor(n_way * n_samples, n_way).zero_()
    for i in range(n_way):
        label[n_samples * i:n_samples * (i + 1), i] = 1
    return to_variable(label)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)