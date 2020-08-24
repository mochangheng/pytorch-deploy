import torch
import torchvision.models as models
from torch_deploy import deploy

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
deploy(resnet18, pre=torch.tensor)