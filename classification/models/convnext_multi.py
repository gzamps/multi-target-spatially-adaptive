import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import dynconv
from models.convnext_util import *

__all__ = ['convnext_tiny_multi', 'convnext_small_multi', 'convnext_base_multi', 'convnext_large_multi', 'convnext_xlarge_multi' ]

class ConvNeXt_multi(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 sparse = False, calibration = False, target_densities = [0.25, 0.5, 0.75], multi_norm = False
                 ):
        super().__init__()
        self.sparse = sparse
        self.target_densities = target_densities

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):

            stage = nn.Sequential(
                *[Block_multi(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, sparse=self.sparse, target_densities=target_densities) for j in range(depths[i])]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):

            # Linear and Conv2D layers with 1 output channel, show problem in trunc_normal_ in 4090
            try:
                # print ("could at" ,m)
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            except:
                # print ("!couldn't at" ,m)
                nn.init.uniform_(m.weight)
                # nn.init.constant_(m.bias, 0)
            # if isinstance(m, nn.Linear):
            #     if m.out_features == 1:
            #         nn.init.uniform_(m.weight)
            #         nn.init.constant_(m.bias, 0)
            #         return 
            # if isinstance(m, nn.Conv2d):
            #     if m.out_channels == 1: 
            #         nn.init.uniform_(m.weight)
            #         nn.init.constant_(m.bias, 0)
            #         return 

            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, meta):
        for i in range(4):
            x = self.downsample_layers[i](x)
            (x, meta) = self.stages[i]( (x,meta) )
        return (self.norm(x.mean([-2, -1])), meta) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, meta=None):
        # print ( meta)
        x, meta = self.forward_features(x, meta)
        x = self.head(x)
        return x, meta

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def initialize_norm_weights_for_multiple_densities( model ):
    r""" Initializes normalization weights for each target budget
     with the normalization weights of the dense model
    """   
    print( "Pass BN weigths also")
    for stage in model.stages:
        for bl in stage:
            print (bl)
            for target_density in model.target_densities:
                norm_module_name = "norm_"+str(target_density).replace('.', '')

                getattr(bl, norm_module_name).weight = bl.norm.weight
                getattr(bl, norm_module_name).bias = bl.norm.bias

                print("passing weight for ", norm_module_name)

    return model

@register_model
def convnext_tiny_multi(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt_multi(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], multi_norm = True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
        model = initialize_norm_weights_for_multiple_densities( model )

    return model

@register_model
def convnext_small_multi(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt_multi(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], multi_norm = True, **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model = initialize_norm_weights_for_multiple_densities( model )

    return model

@register_model
def convnext_base_multi(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_multi(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], multi_norm = True, **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model = initialize_norm_weights_for_multiple_densities( model )

    return model

@register_model
def convnext_large_multi(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_multi(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], multi_norm = True, **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model = initialize_norm_weights_for_multiple_densities( model )

    return model

@register_model
def convnext_xlarge_multi(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_multi(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], multi_norm = True, **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model = initialize_norm_weights_for_multiple_densities( model )

    return model
