import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import dynconv
from models.resnet_util import *

__all__ = [ 'resnet18_multi', 'resnet50_multi', 'resnet101_multi']           


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet_multi(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, sparse=False, target_densities = [0.25, 0.5, 0.75], width_mult=1., **kwargs):
        super(ResNet_multi, self).__init__()
        self.sparse = sparse
        print ( "in constructor ", target_densities)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64*width_mult)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64*width_mult), layers[0], target_densities=target_densities)
        self.layer2 = self._make_layer(block, int(128*width_mult), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], target_densities=target_densities)
        self.layer3 = self._make_layer(block, int(256*width_mult), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], target_densities=target_densities)
        self.layer4 = self._make_layer(block, int(512*width_mult), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], target_densities=target_densities)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*width_mult * block.expansion), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.target_densities = target_densities

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, target_densities=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, sparse=self.sparse, target_densities=target_densities))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, sparse=self.sparse, target_densities=target_densities))

        return nn.Sequential(*layers)

    def forward(self, x, meta=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, meta = self.layer1((x,meta))
        x, meta = self.layer2((x,meta))
        x, meta = self.layer3((x,meta))
        x, meta = self.layer4((x,meta))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, meta

def _resnet_multi(arch, block, layers, pretrained, progress, target_densities = [0.25, 0.5, 0.75], **kwargs):
    model = ResNet_multi(block, layers, target_densities=target_densities,  **kwargs)
    if pretrained:
        print( "Loading pretrained ", arch)
        print ("Before loading bns: ", model.layer1[0].bn1.weight)
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

        for bl in model.layer1:

            for target_density in target_densities:
                module_name1 = "bn1_"+str(target_density).replace('.', '')
                module_name2 = "bn2_"+str(target_density).replace('.', '')
                module_name3 = "bn3_"+str(target_density).replace('.', '')

                getattr(bl, module_name1).weight = bl.bn1.weight
                getattr(bl, module_name2).weight = bl.bn2.weight

                getattr(bl, module_name1).bias = bl.bn1.bias
                getattr(bl, module_name2).bias = bl.bn2.bias

                if isinstance (bl, Bottleneck_multi):
                    getattr(bl, module_name3).weight = bl.bn3.weight
                    getattr(bl, module_name3).bias = bl.bn3.bias

        for bl in model.layer2:
            for target_density in target_densities:
                module_name1 = "bn1_"+str(target_density).replace('.', '')
                module_name2 = "bn2_"+str(target_density).replace('.', '')
                module_name3 = "bn3_"+str(target_density).replace('.', '')

                getattr(bl, module_name1).weight = bl.bn1.weight
                getattr(bl, module_name2).weight = bl.bn2.weight

                getattr(bl, module_name1).bias = bl.bn1.bias
                getattr(bl, module_name2).bias = bl.bn2.bias

                if isinstance (bl, Bottleneck_multi):
                    getattr(bl, module_name3).weight = bl.bn3.weight
                    getattr(bl, module_name3).bias = bl.bn3.bias


        for bl in model.layer3:
            for target_density in target_densities:
                module_name1 = "bn1_"+str(target_density).replace('.', '')
                module_name2 = "bn2_"+str(target_density).replace('.', '')
                module_name3 = "bn3_"+str(target_density).replace('.', '')

                getattr(bl, module_name1).weight = bl.bn1.weight
                getattr(bl, module_name2).weight = bl.bn2.weight

                getattr(bl, module_name1).bias = bl.bn1.bias
                getattr(bl, module_name2).bias = bl.bn2.bias

                if isinstance (bl, Bottleneck_multi):
                    getattr(bl, module_name3).weight = bl.bn3.weight
                    getattr(bl, module_name3).bias = bl.bn3.bias

        for bl in model.layer4:
            for target_density in target_densities:
                module_name1 = "bn1_"+str(target_density).replace('.', '')
                module_name2 = "bn2_"+str(target_density).replace('.', '')
                module_name3 = "bn3_"+str(target_density).replace('.', '')

                getattr(bl, module_name1).weight = bl.bn1.weight
                getattr(bl, module_name2).weight = bl.bn2.weight

                getattr(bl, module_name1).bias = bl.bn1.bias
                getattr(bl, module_name2).bias = bl.bn2.bias

                if isinstance (bl, Bottleneck_multi):
                    getattr(bl, module_name3).weight = bl.bn3.weight
                    getattr(bl, module_name3).bias = bl.bn3.bias    


        print ("After loading bns: ", model.layer1[0].bn1.weight)                                 
    
    return model

def resnet18_multi(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # print ( "in model 1 ", target_densities)
    print('Model: Resnet 18_multi')

    return _resnet_multi('resnet18', BasicBlock_multi, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50_multi(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 50_multi')
    return _resnet_multi('resnet50', Bottleneck_multi, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101_multi(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 101_multi')
    return _resnet_multi('resnet101', Bottleneck_multi, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
