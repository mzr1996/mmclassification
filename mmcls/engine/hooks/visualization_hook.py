# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
from typing import Optional, Sequence

from mmengine import FileClient
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer, WandbVisBackend

from mmcls.registry import HOOKS
from mmcls.structures import ClsDataSample


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Classification Visualization Hook. Used to visualize validation and
    testing prediction results.

    - If ``out_dir`` is specified, all storage backends are ignored
      and save the image to the ``out_dir``.
    - If ``show`` is True, plot the result image in a window, please
      confirm you are able to access the graphical interface.

    Args:
        enable (bool): Whether to enable this hook. Defaults to False.
        interval (int): The interval of samples to visualize. Defaults to 5000.
        show (bool): Whether to display the drawn image. Defaults to False.
        out_dir (str, optional): directory where painted images will be saved
            in the testing process. If None, handle with the backends of the
            visualizer. Defaults to None.
        **kwargs: other keyword arguments of
            :meth:`mmcls.visualization.ClsVisualizer.add_datasample`.
    """

    def __init__(self,
                 enable=False,
                 interval: int = 5000,
                 show: bool = False,
                 out_dir: Optional[str] = None,
                 **kwargs):
        self._visualizer: Visualizer = Visualizer.get_current_instance()

        self.enable = enable
        self.interval = interval
        self.show = show
        self.out_dir = out_dir
        if out_dir is not None:
            self.file_client = FileClient.infer_client(uri=out_dir)
        else:
            self.file_client = None

        self.draw_args = {**kwargs, 'show': show}

    def _draw_samples(self,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[ClsDataSample],
                      step: int = 0) -> None:
        """Visualize every ``self.interval`` samples from a data batch.

        Args:
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ClsDataSample`]): Outputs from model.
            step (int): Global step value to record. Defaults to 0.
        """
        if self.enable is False:
            return

        batch_size = len(data_samples)
        images = data_batch['inputs']
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            image = images[sample_id - start_idx]
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')

            data_sample = data_samples[sample_id - start_idx]
            if 'img_path' in data_sample:
                # osp.basename works on different platforms even file clients.
                sample_name = osp.basename(data_sample.get('img_path'))
            else:
                sample_name = str(sample_id)

            draw_args = self.draw_args
            if self.out_dir is not None:
                draw_args['out_file'] = self.file_client.join_path(
                    self.out_dir, f'{sample_name}_{step}.png')

            self._visualizer.add_datasample(
                sample_name,
                image=image,
                data_sample=data_sample,
                step=step,
                **self.draw_args,
            )

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[ClsDataSample]) -> None:
        """Visualize every ``self.interval`` samples during validation.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ClsDataSample`]): Outputs from model.
        """
        if isinstance(runner.train_loop, EpochBasedTrainLoop):
            step = runner.epoch
        else:
            step = runner.iter

        self._draw_samples(batch_idx, data_batch, outputs, step=step)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[ClsDataSample]) -> None:
        """Visualize every ``self.interval`` samples during test.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): Outputs from model.
        """
        self._draw_samples(batch_idx, data_batch, outputs, step=0)


@HOOKS.register_module()
class WandbVisualizationHook(Hook):
    """Classification Visualization Hook. Used to visualize validation and
    testing prediction results.

    - If ``out_dir`` is specified, all storage backends are ignored
      and save the image to the ``out_dir``.
    - If ``show`` is True, plot the result image in a window, please
      confirm you are able to access the graphical interface.

    Args:
        enable (bool): Whether to enable this hook. Defaults to False.
        interval (int): The interval of samples to visualize. Defaults to 5000.
        show (bool): Whether to display the drawn image. Defaults to False.
        out_dir (str, optional): directory where painted images will be saved
            in the testing process. If None, handle with the backends of the
            visualizer. Defaults to None.
        **kwargs: other keyword arguments of
            :meth:`mmcls.visualization.ClsVisualizer.add_datasample`.
    """

    def __init__(self, enable: bool = False, interval: int = 5000):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.wandb = None
        for backend in Visualizer.get_current_instance()._vis_backends:
            if isinstance(backend, WandbVisBackend):
                self.wandb = backend.experiment

        assert self.wandb is not None, 'No wandb visualization backend.'

        self.enable = enable
        self.interval = interval

    def before_run(self, runner: Runner) -> None:
        val_dataset = runner.val_dataloader.dataset
        self.CLASSES = getattr(val_dataset, 'metainfo',
                               {}).get('classes', None)

        self.data_table = self.wandb.Table(
            columns=['name', 'image', 'ground truth'])
        self._name_idx_map = {}

        data_artifact = self.wandb.Artifact('validation', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

        columns = ['step', 'name', 'image', 'ground truth', 'prediction']
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_image(self, name, image, data_sample):
        if 'gt_label' in data_sample:
            idx = data_sample.gt_label.label.tolist()
            if self.CLASSES is not None:
                gt_labels = [self.CLASSES[i] for i in idx]
            else:
                gt_labels = [str(i) for i in idx]
            gt_label = ', '.join(gt_labels)
        else:
            gt_label = 'N/A'

        self.data_table.add_data(name, self.wandb.Image(image), gt_label)
        idx = len(self._name_idx_map)
        self._name_idx_map[name] = idx
        return idx

    def _add_prediction(self, idx, data_sample, step):
        label_idx = data_sample.pred_label.label.tolist()
        if self.CLASSES is not None:
            pred_labels = [self.CLASSES[i] for i in label_idx]
        else:
            pred_labels = [str(i) for i in label_idx]
        pred_label = ', '.join(pred_labels)

        self.eval_table.add_data(
            step,
            self.data_table_ref.data[idx][0],
            self.data_table_ref.data[idx][1],
            self.data_table_ref.data[idx][2],
            pred_label,
        )

    def _draw_samples(self,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[ClsDataSample],
                      step: int = 0) -> None:
        """Visualize every ``self.interval`` samples from a data batch.

        Args:
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ClsDataSample`]): Outputs from model.
            step (int): Global step value to record. Defaults to 0.
        """
        if self.enable is False:
            return

        batch_size = len(data_samples)
        images = data_batch['inputs']
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            image = images[sample_id - start_idx]
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')

            data_sample = data_samples[sample_id - start_idx]
            if 'img_path' in data_sample:
                # osp.basename works on different platforms even file clients.
                sample_name = osp.basename(data_sample.get('img_path'))
            else:
                sample_name = str(sample_id)

            self._visualizer.add_datasample(
                sample_name,
                image=image,
                data_sample=data_sample,
                step=step,
                **self.draw_args,
            )

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[ClsDataSample]) -> None:
        """Visualize every ``self.interval`` samples during validation.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ClsDataSample`]): Outputs from model.
        """
        if isinstance(runner.train_loop, EpochBasedTrainLoop):
            step = runner.epoch
        else:
            step = runner.iter

        self._draw_samples(batch_idx, data_batch, outputs, step=step)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[ClsDataSample]) -> None:
        """Visualize every ``self.interval`` samples during test.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): Outputs from model.
        """
        self._draw_samples(batch_idx, data_batch, outputs, step=0)
