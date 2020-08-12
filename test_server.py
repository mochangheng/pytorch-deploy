import torch.nn as nn
import torchvision.models as models

from torch_deploy import deploy

def preproc(d):
    return d["array"]
resnet18 = models.resnet18(pretrained=True)
deploy(resnet18, pre=preproc)