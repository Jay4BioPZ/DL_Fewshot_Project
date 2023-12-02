# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        # z_support, z_query = self.parse_feature(x, is_feature)

        # z_support = z_support.contiguous()
        # z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # dists = euclidean_dist(z_query, z_proto)
        # scores = -dists

        z = self.encoder.forward(x)
        zs = z[:n_way * n_shot * self.n_augment]
        zq = z[n_way * n_shot * self.n_augment:]
        # add a column of ones for the bias
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        else:
            wb = self.rr_standard(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        w = wb.narrow(dimension=0, start=0, length=self.output_dim)
        b = wb.narrow(dimension=0, start=self.output_dim, length=1)
        out = mm(zq, w) + b
        y_hat = self.adjust(out)


        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x) # gives Y_hat = f(phi(x';w);w_epsilon)

        return self.loss_fn(scores, y_query ) # gives summation of loss --> L(Y_hat, y') 

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss": avg_loss / float(i + 1)})

def loss(self, sample):
        xs, xq = Variable(sample['xs']), Variable(sample['xq'])
        assert (xs.size(0) == xq.size(0))
        n_way, n_shot, n_query = xs.size(0), xs.size(1), xq.size(1)
        if n_way * n_shot * self.n_augment > self.output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(self.output_dim + 1).cuda())
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(n_way * n_shot * self.n_augment).cuda())

        y_inner = make_float_label(n_way, n_shot * self.n_augment) / np.sqrt(n_way * n_shot * self.n_augment)
        y_outer_binary = make_float_label(n_way, n_query)
        y_outer = make_long_label(n_way, n_query)

        x = torch.cat([xs.view(n_way * n_shot * self.n_augment, *xs.size()[2:]),
                       xq.view(n_way * n_query, *xq.size()[2:])], 0)

        x, y_outer_binary, y_outer = shuffle_queries_multi(x, n_way, n_shot, n_query, self.n_augment, y_outer_binary,
                                                           y_outer)

        # z = self.encoder.forward(x)
        # zs = z[:n_way * n_shot * self.n_augment]
        # zq = z[n_way * n_shot * self.n_augment:]
        # # add a column of ones for the bias
        # ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
        # if rr_type == 'woodbury':
        #     wb = self.rr_woodbury(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        # else:
        #     wb = self.rr_standard(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        # w = wb.narrow(dimension=0, start=0, length=self.output_dim)
        # b = wb.narrow(dimension=0, start=self.output_dim, length=1)
        # out = mm(zq, w) + b
        # y_hat = self.adjust(out)
        # print("%.3f  %.3f  %.3f" % (w.mean()*1e5, b.mean()*1e5, y_hat.max()))

        _, ind_prediction = torch.max(y_hat, 1)
        _, ind_gt = torch.max(y_outer_binary, 1)

        loss_val = self.L(y_hat, y_outer)
        acc_val = torch.eq(ind_prediction, ind_gt).float().mean()
        # print('Loss: %.3f Acc: %.3f' % (loss_val.data[0], acc_val.data[0]))
        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }



def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
