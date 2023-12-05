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
# here import the function from lrd2_labels but not r2d2_labels
from methods.r2d2_utils.lrd2_labels import make_float_label
from methods.r2d2_utils.roll import roll


class LRD2(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(LRD2, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = n_way
        self.n_support = n_support
        self.n_augment = 1
        self.embed_dim = None
        self.learn_lambda = False
        self.init_lambda = 1
        self.lambda_base = 1
        self.linsys = False
        self.lambda_rr = LambdaLayer(self.learn_lambda, self.init_lambda, self.lambda_base)
        self.adjust = AdjustLayer(init_scale=1e-4, base=1)
        # iterations of the IR  
        self.iterations = 5
        
    def set_forward(self, x, is_feature=False):
        '''
        x: [n_way, n_support + n_query, **embed_dim]
        assume that n_way > 2, so that we always need multiple logistic regressions
        '''
        
        # prepare y_support "vector"
        # return an one-hot vector of shape [n_way*n_support]
        y_support = make_float_label(self.n_support, self.n_way*self.n_support*self.n_augment)
        
        # encode feature
        z_support, z_query = self.parse_feature(x, is_feature)
        # of shape [n_way, n_support, **embed_dim] and [n_way, n_query, **embed_dim]
        
        # get the number of embedding dimensions
        self.embed_dim = z_support.size(-1)

        # reslice the tensor to be [n_way*n_support, **embed_dim] and [n_way*n_query, **embed_dim]
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # save n_way scores per query, pick best for each query to know which class it is
        scores = Variable(torch.FloatTensor(self.n_query * self.n_way, self.n_way).zero_().cuda())
        
        # compute score for each class (each column of scores would be a vector of scores for each class)
        for i in range(self.n_way):
            # re-init weight
            w0 = Variable(torch.FloatTensor(self.n_way * self.n_support * self.n_augment).zero_().cuda())
            wb = self.ir_logistic(z_support, w0, y_support)
            y_hat = mm(z_query, wb)
            scores[:, i] = y_hat.squeeze()
            # should the y_support ground truth label to the next class
            y_support = roll(y_support, self.n_support)

        return scores


    def set_forward_loss(self, x):
        # prepare query and support set
        # y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        # y_support = Variable(y_support.cuda()) # of shape [n_way * n_support]
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x) # gives Y_hat = f(phi(x';w);w_epsilon)
        
        # gives summation of loss --> L(Y_hat, y')
        return self.loss_fn(scores, y_query)
      
    def ir_logistic(self, X, w0, y_support):
        '''
        X: z_support
        w0: initial weight
        y_support: one-hot vector
        '''
        # iteration 0
        eta = w0  # + zeros
        mu = torch.sigmoid(eta)
        s = mu * (1 - mu)
        z = eta + (y_support - mu) / s
        S = torch.diag(s)
        # Woodbury with regularization
        w_ = mm(t(X, 0, 1), inv(mm(X, t(X, 0, 1)) + self.lambda_rr(inv(S))))
        z_ = t(z.unsqueeze(0), 0, 1)
        w = mm(w_, z_)
        # it 1...N
        for i in range(self.iterations - 1):
            # eta is w^T x
            eta = w0 + mm(X, w).squeeze(1)
            mu = torch.sigmoid(eta)
            s = mu * (1 - mu)
            z = eta + (y_support - mu) / s
            S = torch.diag(s)
            z_ = t(z.unsqueeze(0), 0, 1)
            if not self.linsys:
                w_ = mm(t(X, 0, 1), inv(mm(X, t(X, 0, 1)) + self.lambda_rr(inv(S))))
                w = mm(w_, z_)
            else:
                A = mm(X, t(X, 0, 1)) + self.lambda_rr(inv(S))
                w_, _ = solve(z_, A)
                w = mm(t(X, 0, 1), w_)

        return w
