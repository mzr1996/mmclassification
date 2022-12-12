# Copyright (c) OpenMMLab. All rights reserved.
import re
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.config import ConfigDict
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmcls.registry import MODELS


def convert_key(key, rules):
    """Convert key name according to a series of rules.

    Every rule is a dict and should have the below keys:

    - **pattern**:
    - **repl**
    """
    for rule in rules:
        assert 'pattern' in rule
        pattern = rule.pattern
        match = re.match(pattern, key)
        if match is None:
            continue
        match_group = match.groups()
        repl = rule.get('repl', None)

        key_action = rule.get('key_action', None)
        if key_action is not None:
            #  key_action = eval(key_action)
            assert callable(key_action)
            match_group = key_action(*match_group)
        start, end = match.span(0)
        if repl is not None:
            key = key[:start] + repl.format(*match_group) + key[end:]
        else:
            for i, sub in enumerate(match_group):
                start, end = match.span(i + 1)
                key = key[:start] + str(sub) + key[end:]
    return key


@MODELS.register_module()
class CompareClassifiers(BaseModel):
    """Compare the output of two network."""

    def __init__(
        self,
        model1: dict,
        model2: dict,
        train_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        map_dict=None,
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
        torch.save(self.model2.state_dict(), 'model2.pth')
        if map_dict is None:
            map_dict = [
                ConfigDict(pattern=key2, repl=key1) for key2, key1 in zip(
                    self.model2.state_dict(), self.model1.state_dict())
            ]
        self.map_dict = map_dict

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
        all_info1, grad_norm1 = self.step_optimizer(optim_wrapper['model1'],
                                                    parsed_losses1,
                                                    self.model1)
        all_info2, grad_norm2 = self.step_optimizer(optim_wrapper['model2'],
                                                    parsed_losses2,
                                                    self.model2.model)
        self._compare_finely(all_info1, all_info2)
        self._compare_parameters(self.model1, self.model2.model)
        logger = MMLogger.get_current_instance()
        # If the grad_norm is slightly different but all grad is the same,
        # it's probably caused by the parameter order. Don't worry.
        logger.info(f'\nloss1: {parsed_losses1}'
                    f'\nloss2: {parsed_losses2}'
                    f'\ngrad_norm1: {grad_norm1.item()}'
                    f'\ngrad_norm2: {grad_norm2.item()}')
        return OrderedDict(
            sorted(list(log_vars1.items()) + list(log_vars2.items())))

    def step_optimizer(self, optimizer: OptimWrapper, loss: torch.Tensor,
                       module: nn.Module):
        optimizer.backward(loss)
        all_info = OrderedDict()
        device = self.data_preprocessor.device
        global_grad_norm = torch.zeros(1, device=device)

        def get_name(src):
            for name, dst in module.named_parameters():
                if src is dst:
                    return name
            raise RuntimeError

        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                global_grad_norm.add_(grad.pow(2).sum())
                all_info[get_name(p)] = {
                    'grad': grad,
                    'optimizer_args':
                    {k: v
                     for k, v in group.items() if k != 'params'}
                }
        global_grad_norm = torch.sqrt(global_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        return all_info, global_grad_norm

    def _compare_finely(self, all_info1, all_info2):
        assert len(all_info1) == len(all_info2)

        all_match = True
        logger = MMLogger.get_current_instance()

        for key2, info2 in all_info2.items():
            grad2 = info2['grad']
            optimizer_args2 = info2['optimizer_args']
            key1 = convert_key(key2, self.map_dict)
            assert key1 in all_info1, f'{key1} not in {list(all_info1.keys())}'
            info1 = all_info1[key1]
            grad1 = info1['grad']
            optimizer_args1 = info1['optimizer_args']
            optimizer_args1.pop('decay_mult', None)
            assert grad1.shape == grad2.shape
            if (grad1 != grad2).any():
                all_match = False
                logger.info(
                    f'The grad of "{key1}" is different between two models.')
            if optimizer_args1 != optimizer_args2:
                all_match = False
                logger.info(
                    f'The optimizer args of "{key1}" is different between two models.'
                )
        if all_match:
            logger.info('All grad match.')

    def _compare_parameters(self, model1, model2):
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        logger = MMLogger.get_current_instance()
        for key2, p2 in reversed(params2.items()):
            key1 = convert_key(key2, self.map_dict)
            p1 = params1[key1]
            if (p1 != p2).any():
                logger.info(
                    f'The parameter of "{key1}" is different between two models.'
                )

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
