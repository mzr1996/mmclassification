# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.optim import OptimWrapper

from mmcls.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TimmOptimizerConstructor:

    def __init__(self, optim_wrapper_cfg: dict, *args, **kwargs) -> None:
        self.optim_wrapper_cfg = optim_wrapper_cfg

    def __call__(self, module: nn.Module):
        from timm.optim import create_optimizer_v2
        optimizer = create_optimizer_v2(module, **self.optim_wrapper_cfg)
        return TimmOptimWrapper(optimizer)


@OPTIM_WRAPPERS.register_module()
class TimmOptimWrapper(OptimWrapper):

    def __init__(self, optimizer, clip_grad=None):
        super().__init__(optimizer=optimizer)

        self.scaler = torch.cuda.amp.GradScaler()
        self.clip_grad = clip_grad
        if self.clip_grad is not None:
            raise NotImplementedError

    def update_params(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        loss = self.scaler.scale(loss)
        self.backward(loss)
        self.scaler.step(self.optimizer)
        self.scaler.update()
