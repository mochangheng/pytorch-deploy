import torch.nn as nn
import torchvision.models as models

from torch_deploy import deploy

resnet18 = models.resnet18(pretrained=True)
deploy(resnet18)

# TODO: Add HTTPS example?