import e2cnn.nn as enn
import torch.nn as nn
import torch
from BaseBlocks import BasicConv2d
from e2cnn import gspaces
import os
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (constant_init, kaiming_init)
# Set default Orientation=8, .i.e, the group C8
# One can change it by passing the env Orientation=xx
Orientation = 8
# keep similar computation or similar params
# One can change it by passing the env fixparams=True
fixparams = False
if 'Orientation' in os.environ:
    Orientation = int(os.environ['Orientation'])
if 'fixparams' in os.environ:
    fixparams = True

gspace = gspaces.Rot2dOnR2(N=Orientation)

def regular_feature_type(gspace: gspaces.GSpace, planes: int):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()
    if fixparams:
        planes *= math.sqrt(N)
    planes = planes / N
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)

FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


def convnxn(inplanes, outplanes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    out_type = FIELD_TYPE['regular'](gspace, outplanes)
    return enn.R2Conv(in_type, out_type, kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      dilation=dilation,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r, )


def ennReLU(inplanes, inplace=True):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.ReLU(in_type, inplace=inplace)


def ennInterpolate(inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)


def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def build_conv_layer(cfg, *args, **kwargs):
    layer = convnxn(*args, **kwargs)
    return layer


def build_norm_layer(cfg, num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


class ReConvModule(enn.EquivariantModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 conv_cfg=None,
                 ):
        super(ReConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        self.in_type = enn.FieldType(gspace, [gspace.regular_repr] * in_channels)
        self.out_type = enn.FieldType(gspace, [gspace.regular_repr] * out_channels)

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        #self.kernel_size = kernel_size
        #self.stride = stride
        #self.padding = padding
        #self.dilation = dilation
        #self.transposed = False
        #self.output_padding = padding
        #self.groups = groups
        # Use msra init by default
        #self.init_weights()

    #def init_weights(self):
    #    nonlinearity = 'relu' if self.activation is None else self.activation
        # kaiming_init(self.conv, nonlinearity=nonlinearity)
        # if self.with_norm:
        #     constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        x = self.conv(x)
        return x

    def evaluate_output_shape(self, input_shape):
        return input_shape

class ReConvBnLelu(enn.EquivariantModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg="FLA",
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ReConvBnLelu, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.in_type = enn.FieldType(gspace, [gspace.regular_repr] * in_channels)
        self.out_type = enn.FieldType(gspace, [gspace.regular_repr] * out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = padding
        self.groups = groups

        # build normalization layers
        if self.with_norm:
            print("with_norm ")
            #while(1):True
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if conv_cfg != None and conv_cfg['type'] == 'ORConv':
                norm_channels = int(norm_channels * 8)
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            print("with_activatation ")
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = ennReLU(out_channels, inplace=self.inplace)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        # kaiming_init(self.conv, nonlinearity=nonlinearity)
        # if self.with_norm:
        #     constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x

    def evaluate_output_shape(self, input_shape):
        return input_shape

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)                                        
#rotation-aware dynamic refinement module
#generate region-aware dynamic filters to refinement 
#https://github.com/lartpang/HDFNet/blob/e2e4136a336f171481d2a6a954e901568932b8d3/module/BaseBlocks.py
#https://github.com/dbbert/dfn/blob/master/layers/dynamic_filter_layer.py
#https://github.com/csuhan/ReDet/blob/master/mmdet/models/backbones/re_resnet.py
#https://github.com/QUVA-Lab/e2cnn/blob/274a9c6a03accdafe3b1d7eda494d8e7f0e0f0a9/e2cnn/nn/modules/r2_conv/r2_transposed_convolution.py
class RADFLayer(nn.Module):
    def __init__(self, 
                 in_yC,
                 in_mC, 
                 out_C,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                ):
        super(RADFLayer, self).__init__()
        self.fuse = nn.Conv2d(in_mC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.m_c = in_mC
        
        if 0:
            self.gernerate_kernel = nn.Sequential(
                nn.Conv2d(in_yC, in_yC, 3, 1, 1),
                DenseLayer(in_yC, in_yC, k=down_factor),
                nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
            )
        #nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1)
        print("in_yC:",in_yC)
        self.gernerate_re_kernel = ReConvModule(
                in_yC,
                self.m_c * self.kernel_size ** 2,
                1,
                conv_cfg=conv_cfg,
                )
        
        self.ConvBnRelu = ReConvBnLelu(
                in_yC,
                self.m_c,
                3,
                stride =1,
                padding =1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='BN', requires_grad=True),
                activation='relu',
                inplace=True            
                )
        
        self.unfold = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)  

    def forward(self, x):
        N, _, xH, xW = x.size()
        #print(x.shape)
        kernel = self.gernerate_re_kernel(x)
        # convert to tensor
        kernel = kernel.tensor
        #kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        kernel = kernel.reshape([N, self.m_c, self.kernel_size ** 2, xH, xW])
        mid_fea = self.ConvBnRelu(x).tensor
        
        unfold_x = self.unfold(mid_fea).reshape([N, self.m_c, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)
#denset layer which combines rich and various receptive fields and generates powerful mixed feature    
#with both spatial structures and appearance details
#just for test


if 0:
    def conv7x7(inplanes, out_planes, stride=2, padding=3, bias=False):
        """7x7 convolution with padding"""
        in_type = enn.FieldType(gspace, inplanes * [gspace.trivial_repr])
        out_type = FIELD_TYPE['regular'](gspace, out_planes)
        return enn.R2Conv(in_type, out_type, 7,
                          stride=stride,
                          padding=padding,
                          bias=bias,
                          sigma=None,
                          frequencies_cutoff=lambda r: 3 * r, )

    model =RADFLayer(256,64,256)
    x= torch.ones([53,3,32,32])
    conv_trasfer = conv7x7(3,256,stride=1)
    in_type = enn.FieldType(gspace, 3 * [gspace.trivial_repr])
    x = enn.GeometricTensor(x, in_type)
    x = conv_trasfer(x)
    c= model(x)
    print(c.shape)