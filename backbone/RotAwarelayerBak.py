import torch.nn as nn
import torch
from BaseBlocks import BasicConv2d
from e2cnn import gspaces

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
                 norm_cfg=None,
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
        """
        构造特征内的密集连接
        """
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
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """RADRM3x3，利用nn.Unfold实现的动态卷积模块
        Args:
            in_xC (int): 第一个输入的通道数 fea: x
            in_yC (int): 第二个输入的通道数 fea: y generate kernel 
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(RADRM3x3, self).__init__()
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.middle_channel = 64
        if 0:
            self.gernerate_kernel = nn.Sequential(
                nn.Conv2d(in_yC, in_yC, 3, 1, 1),
                DenseLayer(in_yC, in_yC, k=down_factor),
                nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
            )
        #nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1)
        self.gernerate_re_kernel = ReConvModule(
                in_channels,
                out_channels,
                3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
        
        self.ConvBnRelu = ReConvBnLelu(
                in_channels,
                out_channels,
                1,
                stride =1,
                padding =1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
        
        self.unfold = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)
#denset layer which combines rich and various receptive fields and generates powerful mixed feature    
#with both spatial structures and appearance details
#just for test
if 0:
    model =RADRM3x3(10,4,2)
    x= torch.ones([53,10,32,32])
    y= torch.ones([53,4,32,32])
    c= model(x,y)
print(c.shape)