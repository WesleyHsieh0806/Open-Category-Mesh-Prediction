from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torch
from torch import nn 

from torchvision.models.feature_extraction import create_feature_extractor


__all__ = ["get_backbone"]

class CustomRenset(nn.Module):
    def __init__(self, model):
        super(CustomRenset, self).__init__()
        self.model = model
        self.body = create_feature_extractor(
            self.model, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})

    def forward(self, image):
        '''
        * Input: image of shape (B, 3, H, W)
        * Output:
            a dict contains the output of layer conv2_3, conv3_4, conv4_6, conv5_3
            {
                0: tensor of shape(B, 256, H/4, W/4)
                1: tensor of shape(B, 512, H/8, W/8)
                2: tensor of shape(B, 1024, H/16, W/16)
                3: tensor of shape(B, 2048, H/32, W/32)
            }
        '''

        return self.body(image)


def _get_resnet50(pretrained):
    model = resnet50(weights=ResNet50_Weights.DEFAULT) if pretrained else resnet50()
    return CustomRenset(model)


def _get_resnet101(pretrained):
    model = resnet101(weights=ResNet101_Weights.DEFAULT) if pretrained else resnet101()
    return CustomRenset(model)

def _get_resnet152(pretrained):
    model = resnet152(weights=ResNet152_Weights.DEFAULT) if pretrained else resnet152()
    return CustomRenset(model)

def get_backbone(backbone, pretrained=True):
    get_pretrained_backbone = {
        "resnet50": lambda pretrained: _get_resnet50(pretrained),
        "resnet101": lambda pretrained: _get_resnet101(pretrained),
        "resnet152": lambda pretrained: _get_resnet152(pretrained),
    }
    return get_pretrained_backbone[backbone](pretrained)

if __name__ == "__main__":
    model = get_backbone("resnet50", True)
    print(model)