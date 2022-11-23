# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import re

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.config import ConfigDict

from mmcls.registry import MODELS

map_dict = [
    ConfigDict(pattern=r'^cls_token',
         repl=r'backbone.cls_token'),
    ConfigDict(pattern=r'^pos_embed',
         repl=r'backbone.pos_embed'),
    ConfigDict(pattern=r'^patch_embed.proj', repl=r'backbone.patch_embed.projection'),
    ConfigDict(pattern=r'blocks.(\d+).gamma_1',
         repl=r'backbone.layers.{0}.attn.gamma1.weight'),
    ConfigDict(pattern=r'blocks.(\d+).gamma_2',
         repl=r'backbone.layers.{0}.ffn.gamma2.weight'),
    ConfigDict(pattern=r'blocks.(\d+).norm1',
         repl=r'backbone.layers.{0}.ln1'),
    ConfigDict(pattern=r'blocks.(\d+).norm2',
         repl=r'backbone.layers.{0}.ln2'),
    ConfigDict(pattern=r'blocks.(\d+).attn',
         repl=r'backbone.layers.{0}.attn'),
    ConfigDict(pattern=r'blocks.(\d+).mlp.fc1',
         repl=r'backbone.layers.{0}.ffn.layers.0.0'),
    ConfigDict(pattern=r'blocks.(\d+).mlp.fc2',
         repl=r'backbone.layers.{0}.ffn.layers.1'),
    ConfigDict(pattern=r'^norm', repl=r'backbone.ln1'),
    ConfigDict(pattern=r'^head', repl=r'head.layers.head'),
]


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
        torch.save(self.model2.state_dict(), 'model2.pth')

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
        self._compare_finely(all_info1, all_info2, map_dict)
        logger = MMLogger.get_current_instance()
        # If the grad_norm is slightly different but all grad is the same,
        # it's probabily caused by the parameter order. Don't worry.
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

    def _compare_finely(self, all_info1, all_info2, match_mapping):
        assert len(all_info1) == len(all_info2)

        all_match = True
        logger = MMLogger.get_current_instance()

        def convert_key(key):
            for item in match_mapping:
                assert 'pattern' in item
                pattern = item.pattern
                match = re.match(pattern, key)
                if match is None:
                    continue
                match_group = match.groups()
                repl = item.get('repl', None)

                key_action = item.get('key_action', None)
                if key_action is not None:
                    #  key_action = eval(key_action)
                    assert callable(key_action)
                    match_group = key_action(*match_group)
                start, end = match.span(0)
                if repl is not None:
                    key = key[:start] + repl.format(*match_group) + key[end:]
                else:
                    for i, sub in enumerate(match_group):
                        start, end = match.span(i+1)
                        key = key[:start] + str(sub) + key[end:]
            return key

        for key2, info2 in all_info2.items():
            grad2 = info2['grad']
            optimizer_args2 = info2['optimizer_args']
            key1 = convert_key(key2)
            assert key1 in all_info1
            info1 = all_info1[key1]
            grad1 = info1['grad']
            optimizer_args1 = info1['optimizer_args']
            optimizer_args1.pop('decay_mult', None)
            assert grad1.shape == grad2.shape
            if (grad1 != grad2).any():
                all_match = False
                logger.info(
                    f'The grad of "{key1}" if different between two models.')
            if optimizer_args1 != optimizer_args2:
                all_match = False
                logger.info(
                    f'The grad of "{key1}" if different between two models.')
        if all_match:
            logger.info('All grad match.')


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
