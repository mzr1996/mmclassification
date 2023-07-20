from mmengine.config import read_base
from models.example_net import ExampleNet

with read_base():
    from mmpretrain.configs.resnet.resnet18_8xb32_in1k import *
    from torch.nn import Linear

model.backbone.type = ExampleNet
model.backbone.depth = 18
model.head.in_channels = 512
