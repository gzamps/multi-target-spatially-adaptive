import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import dynconv
import models.resnet_util


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse

        if sparse:
            # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
            self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1)

        self.fast = False

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
        else:
            assert meta is not None

            m = self.masker(x, meta)
            
            mask_dilate, mask = m['dilate'], m['std'] 

            x = dynconv.conv3x3(self.conv1, x, None, mask_dilate)
            x = dynconv.bn_relu(self.bn1, self.relu, x, mask_dilate)
            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, None, x, mask)

            out = identity + dynconv.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' + 
              f'oup {planes * self.expansion}, stride {stride}')

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)        
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.fast = True

        if sparse:
            self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=stride)

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out += identity
        else:
            assert meta is not None

            m = self.masker(x, meta)
            
            mask_dilate, mask = m['dilate'], m['std']

            x = dynconv.conv1x1(self.conv1, x, mask_dilate)
            x = dynconv.bn_relu(self.bn1, self.relu, x, mask_dilate)
            # x = self.in1(x)
            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, self.relu, x, mask)
            # x = self.in2(x)        
            x = dynconv.conv1x1(self.conv3, x, mask)
            x = dynconv.bn_relu(self.bn3, None, x, mask)
            # x = self.in3(x)
            out = identity + dynconv.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta


class BasicBlock_multi(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False,
                 target_densities=None):
        super(BasicBlock_multi, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse

        if sparse:
             for target_density in target_densities:

                module_name = "masker"+str(target_density).replace('.', '')
                setattr(self, module_name, dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1).cuda())

                module_name1 = "bn1_"+str(target_density).replace('.', '')
                setattr(self, module_name1, nn.BatchNorm2d(planes).cuda())

                module_name2 = "bn2_"+str(target_density).replace('.', '')
                setattr(self, module_name2, nn.BatchNorm2d(planes).cuda())

        self.fast = False
        self.also_scale_inference = False
    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
        else:
            assert meta is not None

            # masker = self.maskers[str(meta['target_density'])]
            # m = masker(x, meta)
            # m = self.maskers[str(meta['target_density'])](x,meta)
            module_name = "masker"+str(meta['target_density']).replace('.', '')
            m = getattr(self, module_name)(x, meta)

            module_name1 = "bn1_"+str(meta['target_density']).replace('.', '')
            module_name2 = "bn2_"+str(meta['target_density']).replace('.', '')
            
            mask_dilate, mask = m['dilate'], m['std'] 

            x = dynconv.conv3x3(self.conv1, x, None, mask_dilate)
            # x = dynconv.bn_relu(self.bn1, self.relu, x, mask_dilate)
            x = dynconv.bn_relu(getattr(self, module_name1), self.relu, x, mask_dilate)

            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            # x = dynconv.bn_relu(self.bn2, None, x, mask)
            x = dynconv.bn_relu(getattr(self, module_name2), None, x, mask)

            out = identity + dynconv.apply_mask(x, mask)
            # if self.also_scale_inference is True:
            # module_name = "scalerfactor"+str(meta['target_density']).replace('.', '')
            # out = identity + dynconv.apply_mask_scaled_learnable(x, mask, getattr(self, module_name), True)
            # print(module_name, ": ", getattr(self, module_name))

        out = self.relu(out)
        return out, meta


class Bottleneck_multi(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False,
                 target_densities=None):
        super(Bottleneck_multi, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' + 
              f'oup {planes * self.expansion}, stride {stride}')

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)        
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse

        if sparse:

            for target_density in target_densities:
 
                module_name = "masker"+str(target_density).replace('.', '')
                setattr(self, module_name, dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1).cuda())

                module_name1 = "bn1_"+str(target_density).replace('.', '')
                setattr(self, module_name1, nn.BatchNorm2d(width).cuda())

                module_name2 = "bn2_"+str(target_density).replace('.', '')
                setattr(self, module_name2, nn.BatchNorm2d(width).cuda())

                module_name3 = "bn3_"+str(target_density).replace('.', '')
                setattr(self, module_name3, nn.BatchNorm2d(planes * self.expansion).cuda())                

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out += identity
        else:
            assert meta is not None

            # masker = self.maskers[str(meta['target_density'])]
            # m = masker(x, meta)
            # m = self.maskers[str(meta['target_density'])](x,meta)
            module_name = "masker"+str(meta['target_density']).replace('.', '')
            m = getattr(self, module_name)(x, meta)

            module_name1 = "bn1_"+str(meta['target_density']).replace('.', '')
            module_name2 = "bn2_"+str(meta['target_density']).replace('.', '')
            module_name3 = "bn3_"+str(meta['target_density']).replace('.', '')

            mask_dilate, mask = m['dilate'], m['std']

            x = dynconv.conv1x1(self.conv1, x, mask_dilate)
            x = dynconv.bn_relu(getattr(self, module_name1), self.relu, x, mask_dilate)
            # x = self.in1(x)
            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.bn_relu(getattr(self, module_name2), self.relu, x, mask)
            # x = self.in2(x)        
            x = dynconv.conv1x1(self.conv3, x, mask)
            x = dynconv.bn_relu(getattr(self, module_name3), None, x, mask)
            # x = self.in3(x)
            out = identity + dynconv.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta
