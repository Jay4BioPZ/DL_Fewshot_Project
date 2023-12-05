import torch
from torch.autograd import Variable

def make_float_label(true_samples, n_samples):
    label = torch.FloatTensor(n_samples).zero_()
    label[0:true_samples] = 1
    return to_variable(label)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)