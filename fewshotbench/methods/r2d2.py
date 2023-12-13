# This code is modified from https://github.com/bertinetto/r2d2
# check config here https://github.com/bertinetto/r2d2/blob/fc0c13ec991bb9f84395cb12de57cb150ce76f8d/scripts/train/conf/fewshots.yaml#L56

import numpy as np
import torch
import torch.nn as nn
from torch import transpose as t
from torch import inverse as inv
from torch import mm
from torch import solve # gesv is deprecated, use solve instead
from torch.autograd import Variable
import wandb

from methods.meta_template import MetaTemplate
from methods.r2d2_utils.adjust import AdjustLayer, LambdaLayer
from methods.r2d2_utils.r2d2_labels import make_float_label


class R2D2(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(R2D2, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = n_way
        self.n_support = n_support
        self.n_augment = 1
        self.rr_type = None
        self.embed_dim = None
        self.learn_lambda = False
        self.init_lambda = 0.1
        self.lambda_base = 1
        self.linsys = False
        self.lambda_rr = LambdaLayer(self.learn_lambda, self.init_lambda, self.lambda_base)
        self.adjust = AdjustLayer(init_scale=1e-4, base=1)
        
    def set_rr_type(self):
        if self.n_way * self.n_support * self.n_augment > self.embed_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(self.embed_dim + 1).cuda())
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(self.n_way * self.n_support * self.n_augment).cuda())
        return rr_type, I

    def set_forward(self, x, is_feature=False):
        '''
        x: [n_way, n_support + n_query, **embed_dim]
        '''
        
        # prepare y_support matrix
        # one-hot encode the ground truth labels into a matrix
        # each column is a one-hot vector for a class
        y_support = make_float_label(self.n_way, self.n_support * self.n_augment) / np.sqrt(self.n_way * self.n_support * self.n_augment)
        
        # encode feature
        z_support, z_query = self.parse_feature(x, is_feature)
        # of shape [n_way, n_support, **embed_dim] and [n_way, n_query, **embed_dim]
        
        # get the number of embedding dimensions
        self.embed_dim = z_support.size(-1)
        # set rr_type
        self.rr_type, I = self.set_rr_type()

        # reslice the tensor to be [n_way*n_support, **embed_dim] and [n_way*n_query, **embed_dim]
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # add a column of ones for the bias
        ones = Variable(torch.unsqueeze(torch.ones(z_support.size(0)).cuda(), 1))
        
        # compute episode-dependent weights through ridge regression
        if self.rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((z_support, ones), 1), self.n_way, self.n_support, I, y_support, self.linsys)

        else:
            wb = self.rr_standard(torch.cat((z_support, ones), 1), self.n_way, self.n_support, I, y_support, self.linsys)

        # extract weights and bias
        # put the name of the input
        w = wb.narrow(0, 0, self.embed_dim) # of shape [**embed_dim, n_way]
        b = wb.narrow(0, self.embed_dim, 1) # of shape [1, n_way]
        
        out = mm(z_query, w) + b # of shape [n_way*n_query, n_way]
        y_hat = self.adjust(out)

        return y_hat


    def set_forward_loss(self, x):
        # prepare query and support set
        # y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        # y_support = Variable(y_support.cuda()) # of shape [n_way * n_support]
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        yhat = self.set_forward(x) # gives Y_hat = f(phi(x';w);w_epsilon)
        
        # gives summation of loss --> L(Y_hat, y')
        return self.loss_fn(yhat, y_query)
      
    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)

        if not linsys:
            w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)
        else:
            A = mm(t(x, 0, 1), x) + self.lambda_rr(I)
            v = mm(t(x, 0, 1), yrr_binary)
            w, _ = solve(v, A)

        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)

        if not linsys:
            w = mm(mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary)
        else:
            A = mm(x, t(x, 0, 1)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = solve(v, A)
            w = mm(t(x, 0, 1), w_)

        return w
