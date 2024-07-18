import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import dynconv

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.sparse = sparse

        if sparse:
            # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
            # self.masker = dynconv.MaskUnitConvNeXt(channels=dim, stride=1, dilate_stride=1)
            self.masker = dynconv.MaskUnit(channels=dim, stride=1, dilate_stride=1)


    def forward(self, xmeta):
        x, meta = xmeta
        # print ( "input: " , x.shape)
        input = x

        if not self.sparse:
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)


            # print ( "out before add: " , x.shape)

            x = input + self.drop_path(x)

        else:
            assert meta is not None
            
            m = self.masker(x, meta)
            
            mask_dilate, mask = m['dilate'], m['std'] 

            # x = dynconv.conv3x3_dw(self.dwconv, x, None, mask_dilate)
            # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            # x = dynconv.bn_relu(self.norm, None, x, mask_dilate)
            # x = dynconv.pwconv(self.pwconv1, x, mask)
            # x = dynconv.bn_relu(self.act, None, x, mask)
            # x = dynconv.pwconv(self.pwconv2, x, mask)


            # assume they are normal stuff: do only the depthwise
            x = dynconv.conv3x3_dw(self.dwconv, x, None, mask)
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = dynconv.bn_relu(self.norm, None, x, mask)
            x = dynconv.pwconv(self.pwconv1, x, mask)
            x = dynconv.bn_relu(self.act, None, x, mask)
            x = dynconv.pwconv(self.pwconv2, x, mask)


            # x = self.pwconv1(x)
            # x = self.act(x)
            # x = self.pwconv2(x)

            # x = self.in1(x)

            # x = dynconv.conv1x1(self.conv2, x, mask_dilate, mask)
            # x = dynconv.bn_relu(self.bn2, None, x, mask)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            # x = self.in2(x)
            # print ( "mask: " , mask.hard.shape)
            # print ( "out before add: " , x.shape)


            x = input + dynconv.apply_mask(self.drop_path(x), mask)

        return (x, meta)

class Block_multi(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=False, target_densities=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.sparse = sparse

        # if sparse:
        #     # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
        #     # self.masker = dynconv.MaskUnitConvNeXt(channels=dim, stride=1, dilate_stride=1)
        #     self.masker = dynconv.MaskUnit(channels=dim, stride=1, dilate_stride=1)
        if sparse:
            # self.maskers = {}
            # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
            # self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
            # self.universaltolocalmask = dynconv.UniversalMaskToLayerV2(stride=self.stride)
            for target_density in target_densities:
                # print  (target_density)

                # self.maskers[str(target_density)] = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1).cuda()
                # self.maskers[str(target_density)] = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1).cuda()
                module_name = "masker"+str(target_density).replace('.', '')
                setattr(self, module_name, dynconv.MaskUnit(channels=dim, stride=1, dilate_stride=1).cuda())            

                norm_module_name = "norm_"+str(target_density).replace('.', '')
                setattr(self, norm_module_name, LayerNorm(dim, eps=1e-6).cuda())

    def forward(self, xmeta):
        x, meta = xmeta
        # print ( "input: " , x.shape)
        input = x

        if not self.sparse:
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            x = input + self.drop_path(x)

        else:
            assert meta is not None
            
            # masker = self.maskers[str(meta['target_density'])]
            # m = masker(x, meta)
            # m = self.maskers[str(meta['target_density'])](x,meta)
            module_name = "masker"+str(meta['target_density']).replace('.', '')
            m = getattr(self, module_name)(x, meta)

            norm_module_name = "norm_"+str(meta['target_density']).replace('.', '')
            
            # m = self.masker(x, meta)
            
            mask_dilate, mask = m['dilate'], m['std'] 

            # x = dynconv.conv3x3_dw(self.dwconv, x, None, mask_dilate)
            # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            # x = dynconv.bn_relu(self.norm, None, x, mask_dilate)
            # x = dynconv.pwconv(self.pwconv1, x, mask)
            # x = dynconv.bn_relu(self.act, None, x, mask)
            # x = dynconv.pwconv(self.pwconv2, x, mask)

            
            # assume they are normal stuff: do only the depthwise
            x = dynconv.conv3x3_dw(self.dwconv, x, None, mask)
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = dynconv.bn_relu(getattr(self, norm_module_name), None, x, mask)
            # x = dynconv.bn_relu(self.norm, None, x, mask)

            x = dynconv.pwconv(self.pwconv1, x, mask)
            x = dynconv.bn_relu(self.act, None, x, mask)
            x = dynconv.pwconv(self.pwconv2, x, mask)


            # x = self.pwconv1(x)
            # x = self.act(x)
            # x = self.pwconv2(x)

            # x = self.in1(x)

            # x = dynconv.conv1x1(self.conv2, x, mask_dilate, mask)
            # x = dynconv.bn_relu(self.bn2, None, x, mask)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            # x = self.in2(x)
            # print ( "mask: " , mask.hard.shape)
            # print ( "out before add: " , x.shape)


            x = input + dynconv.apply_mask(self.drop_path(x), mask)

            # out = self.inout(out)
            # print ( "Doing this")

            # x = input + self.drop_path(x)
        return (x, meta)