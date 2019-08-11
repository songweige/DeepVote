import torch
import torch.nn as nn
from torch.autograd import Variable


## 2D convolution layers
class conv2d(nn.Module):
    def __init__(self, n_pairs, batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
        super(conv2d, self).__init__()
        if batch_norm:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.2,inplace=True),
            )
        else:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.2,inplace=True),
            )
        self.n_pairs = n_pairs
    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x_conv = self.conv_layer(x)
        # print(x_conv.shape)
        x_conv = x_conv.view(-1, self.n_pairs, x_conv.size(1), x_conv.size(2), x_conv.size(3))
        return x_conv

class PermEqui_mean(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_mean, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    # self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.mean(1, keepdim=True).repeat(1, self.n_pairs, 1, 1, 1)
    # xm = self.Lambda.forward(xm)
    # x = self.Gamma.forward(x)
    x = self.Gamma.forward(x-xm)
    # x = x - xm
    return x

class PermEqui_mean2(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_mean2, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.mean(1, keepdim=True).repeat(1, self.n_pairs, 1, 1, 1)
    xm = self.Lambda.forward(xm)
    x = self.Gamma.forward(x)
    # x = self.Gamma.forward(x-xm)
    x = x - xm
    return x

class PermEqui_median(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_median, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    # self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.median(1, keepdim=True)[0].repeat(1, self.n_pairs, 1, 1, 1)
    # xm = self.Lambda.forward(xm)
    # x = self.Gamma.forward(x)
    x = self.Gamma.forward(x-xm)
    # x = x - xm
    return x

class PermEqui_max(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_max, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    # self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.max(1, keepdim=True)[0].repeat(1, self.n_pairs, 1, 1, 1)
    # xm = self.Lambda.forward(xm)
    # x = self.Gamma.forward(x)
    x = self.Gamma.forward(x-xm)
    # x = x - xm
    return x

class DeepVote_mean(nn.Module):
    """docstring for DeepVote_mean"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote_mean, self).__init__()
        self.model = nn.Sequential(
            PermEqui_mean(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_mean(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_mean(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_mean(n_pairs, in_dim=8, out_dim=1,), 
        )
        self.residual = residual
    def forward(self, img):
        if self.residual:
            return self.model(img).squeeze_().mean(1) + img[:, :, 1, :, :].mean(1)
        else:
            return self.model(img).squeeze_().mean(1)

class DeepVote_mean2(nn.Module):
    """docstring for DeepVote_mean2"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote_mean2, self).__init__()
        self.model = nn.Sequential(
            PermEqui_mean2(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_mean2(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_mean2(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_mean2(n_pairs, in_dim=8, out_dim=1,), 
        )
        self.residual = residual
    def forward(self, img):
        if self.residual:
            return self.model(img).squeeze_().mean(1) + img[:, :, 1, :, :].mean(1)
        else:
            return self.model(img).squeeze_().mean(1)

class DeepVote_median(nn.Module):
    """docstring for DeepVote_median"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote_median, self).__init__()
        self.model = nn.Sequential(
            PermEqui_median(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_median(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_median(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_median(n_pairs, in_dim=8, out_dim=1,), 
        )
        self.residual = residual
    def forward(self, img):
        if self.residual:
            return self.model(img).squeeze_().median(1)[0] + img[:, :, 1, :, :].median(1)
        else:
            return self.model(img).squeeze_().median(1)[0]

class DeepVote_max(nn.Module):
    """docstring for DeepVote_max"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote_max, self).__init__()
        self.model = nn.Sequential(
            PermEqui_max(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_max(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_max(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_max(n_pairs, in_dim=8, out_dim=1,), 
        )
        self.residual = residual
    def forward(self, img):
        if self.residual:
            return self.model(img).squeeze_().max(1)[0] + img[:, :, 1, :, :].mean(1)
        else:
            return self.model(img).squeeze_().max(1)[0]

class DeepVote_large(nn.Module):
    """docstring for DeepVote_large"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote_large, self).__init__()
        self.model = nn.Sequential(
            conv2d(n_pairs, batch_norm=False, in_planes=2, out_planes=16, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=16, out_planes=32, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=32, out_planes=16, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=16, out_planes=1, kernel_size=5), # predict the height for each pixel
        )
        self.residual = residual
    def forward(self, img):
        if self.residual:
            return self.model(img).squeeze_().mean(1) + img[:, :, 1, :, :].mean(1)
        else:
            return self.model(img).squeeze_().mean(1)

class DeepVote_weighted(nn.Module):
    """docstring for DeepVote_weighted"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote_weighted, self).__init__()
        self.model = nn.Sequential(
            conv2d(n_pairs, batch_norm=False, in_planes=2, out_planes=8, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=8, out_planes=16, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=16, out_planes=8, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=8, out_planes=2, kernel_size=5), # predict the height for each pixel
        )
        self.residual = residual
    def forward(self, img):
        # B x P x 2 x H x W
        preds = self.model(img)
        preds = nn.functional.softmax(preds[:, :, 0, :, :], 1)*preds[:, :, 1, :, :]
        if self.residual:
            return preds.squeeze_().sum(1) + img[:, :, 1, :, :].mean(1)
        else:
            return preds.squeeze_().sum(1)


class DeepVote(nn.Module):
    """docstring for DeepVote"""
    def __init__(self, n_pairs, residual=False):
        super(DeepVote, self).__init__()
        self.model = nn.Sequential(
            conv2d(n_pairs, batch_norm=False, in_planes=2, out_planes=8, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=8, out_planes=16, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=16, out_planes=8, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=8, out_planes=1, kernel_size=5), # predict the height for each pixel
        )
        self.residual = residual
    def forward(self, img):
        if self.residual:
            return self.model(img).squeeze_().mean(1) + img[:, :, 1, :, :].mean(1)
        else:
            return self.model(img).squeeze_().mean(1)


nets_map = {
    'base': DeepVote,
    'large': DeepVote_large,
    'weighted': DeepVote_weighted,
    'mean': DeepVote_mean,
    'mean2': DeepVote_mean2,
    'median': DeepVote_median,
    'max': DeepVote_max
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn