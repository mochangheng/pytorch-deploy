from functools import partial

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch_deploy import deploy
from torch_deploy.processing import list2PIL
from PIL import Image
import numpy as np

def test1():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()
    deploy(resnet18, pre=torch.tensor)

def test2():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()
    pre = [
        list2PIL,
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        partial(torch.unsqueeze, dim=0)
    ]
    deploy(resnet18, pre=pre)

if __name__ == '__main__':
    test2()
