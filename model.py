import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.autograd import Function



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class HoPRA(nn.Module):

    def __init__(self, input_feat_dim_s, input_feat_dim_t, hidden_dim1, hidden_dim2, nclass):

        super(HoPRA, self).__init__()
        self.s_hcn1 = GraphConvolution(input_feat_dim_s, hidden_dim1)
        self.s_hcn2 = GraphConvolution(hidden_dim1, hidden_dim2)

        self.t_hcn1 = GraphConvolution(input_feat_dim_t, hidden_dim1)
        self.t_hcn2 = GraphConvolution(hidden_dim1, hidden_dim2)

        self.s_gcn1 = GraphConvolution(input_feat_dim_s, hidden_dim1)
        self.s_gcn2 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.s_gcn3 = GraphConvolution(hidden_dim1, hidden_dim2)

        self.t_gcn1 = GraphConvolution(input_feat_dim_t, hidden_dim1)
        self.t_gcn2 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.t_gcn3 = GraphConvolution(hidden_dim1, hidden_dim2)

        self.discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(hidden_dim2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.dc = InnerProductDecoder(0)

        self.Classifier = nn.Linear(hidden_dim2, nclass)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, H, Ht, xs, xtl, xt, ps, pt, t_test_idx):

        'hypergraph encoder'
        # layer 1
        xs1 = self.s_hcn1(xs, H)
        xs1 = F.relu(xs1)
        # layer 2
        xs2 = self.s_hcn2(xs1, H)
        xs2 = F.relu(xs2)

        # layer 1
        xtl1 = self.t_hcn1(xtl, Ht)
        xtl1 = F.relu(xtl1)
        # layer 2
        xtl2 = self.t_hcn2(xtl1, Ht)
        xtl2 = F.relu(xtl2)

        'source network variational graph auto-encoder'
        xs_p1 = self.s_gcn1(xs, ps)
        xs_p1 = F.relu(xs_p1)
        xs_p2 = self.s_gcn2(xs_p1, ps)
        xs_p2 = F.relu(xs_p2)
        xs_p3 = self.s_gcn3(xs_p1, ps)
        xs_p3 = F.relu(xs_p3)
        zs = self.reparameterize(xs_p2, xs_p3)
        ps_hat = self.dc(zs)

        'target network variational graph auto-encoder'
        xt_p1 = self.t_gcn1(xt, pt)
        xt_p1 = F.relu(xt_p1)
        xt_p2 = self.t_gcn2(xt_p1, pt)
        xt_p2 = F.relu(xt_p2)
        xt_p3 = self.t_gcn3(xt_p1, pt)
        xt_p3 = F.relu(xt_p3)
        zt = self.reparameterize(xt_p2, xt_p3)
        pt_hat = self.dc(zt)

        'domain classifier'
        z = self.discriminator(torch.cat((xs2, xtl2)))

        'hypergraph decoder'
        H_hat = torch.sigmoid(torch.mm(xs2, xtl2.T))

        'node classifier'
        x_all = torch.cat((xs2, xtl2, xt_p2[t_test_idx]), 0)
        x_output = self.Classifier(x_all)
        x_output = F.log_softmax(x_output, dim=1)

        return H_hat, x_output, xs2, xtl2, xs_p2, xs_p3, ps_hat, xt_p2, xt_p3, pt_hat, z