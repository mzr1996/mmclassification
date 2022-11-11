# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmcls.registry import MODELS


@MODELS.register_module()
class CompareClassifiers(BaseModel):
    """Compare the output of two network."""

    def __init__(
        self,
        model1: dict,
        model2: dict,
        train_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
    ):
        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super().__init__(data_preprocessor=data_preprocessor)

        if isinstance(model1, dict):
            self.model1 = MODELS.build(model1)
        else:
            self.model1 = model1
        if isinstance(model2, dict):
            self.model2 = MODELS.build(model2)
        else:
            self.model2 = model2

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        data = self.data_preprocessor(data, True)
        with optim_wrapper['model1'].optim_context(self):
            losses1 = self._run_forward(
                self.model1, data, mode='loss')  # type: ignore
        with optim_wrapper['model2'].optim_context(self):
            losses2 = self._run_forward(
                self.model2, data, mode='loss')  # type: ignore
        parsed_losses1, log_vars1 = self.parse_losses(losses1)  # type: ignore
        parsed_losses2, log_vars2 = self.parse_losses(losses2)  # type: ignore
        grad_norm1 = self.step_optimizer(optim_wrapper['model1'],
                                         parsed_losses1)
        grad_norm2 = self.step_optimizer(optim_wrapper['model2'],
                                         parsed_losses2)
        logger = MMLogger.get_current_instance()
        logger.info(f'\nloss1: {parsed_losses1}'
                    f'\nloss2: {parsed_losses2}'
                    f'\ngrad_norm1: {grad_norm1.item()}'
                    f'\ngrad_norm2: {grad_norm2.item()}')
        return OrderedDict(
            sorted(list(log_vars1.items()) + list(log_vars2.items())))

    def step_optimizer(self, optimizer: OptimWrapper, loss: torch.Tensor):
        optimizer.backward(loss)
        device = self.data_preprocessor.device
        global_grad_norm = torch.zeros(1, device=device)
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                global_grad_norm.add_(grad.pow(2).sum())
        global_grad_norm = torch.sqrt(global_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        return global_grad_norm

    def _run_forward(self, model: nn.Module, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = model(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = model(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def parse_losses(
        self,
        losses: Dict[str, torch.Tensor],
        postfix: str = '',
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss, log_vars = super().parse_losses(losses)
        new_log_vars = OrderedDict()
        for k, v in log_vars.items():
            new_log_vars[k + postfix] = v
        return loss, new_log_vars

    def forward(self, inputs):
        raise NotImplementedError
