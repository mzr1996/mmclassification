# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from os import PathLike
from typing import Optional, Sequence, Union

import mmengine
from mmengine.dataset import BaseDataset

from mmcls.registry import DATASETS, TRANSFORMS


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


def isabs(uri):
    return osp.isabs(uri) or ('://' in uri)


@DATASETS.register_module()
class MultiTaskDataset(BaseDataset):
    """Custom dataset for multi-task dataset.

    To use the dataset, please generate and provide an annotation file in the
    below format:

    .. code-block:: json

        {
          "metainfo": {
            "tasks":
              [
              'gender'
              'wear'
              ]
          },
          "data_list": [
            {
              "img_path": "a.jpg",
              gt_label:{
                  "gender": 0,
                  "wear": [1, 0, 1, 0]
                }
            },
            {
              "img_path": "b.jpg",
              gt_label:{
                  "gender": 1,
                  "wear": [1, 0, 1, 0]
                }
            }
          ]
        }

    Assume we put our dataset in the ``data/mydataset`` folder in the
    repository and organize it as the below format: ::

        mmclassification/
        └── data
            └── mydataset
                ├── annotation
                │   ├── train.json
                │   ├── test.json
                │   └── val.json
                ├── train
                │   ├── a.jpg
                │   └── ...
                ├── test
                │   ├── b.jpg
                │   └── ...
                └── val
                    ├── c.jpg
                    └── ...

    We can use the below config to build datasets:

    .. code:: python

        >>> from mmcls.datasets import build_dataset
        >>> train_cfg = dict(
        ...     type="MultiTaskDataset",
        ...     ann_file="annotation/train.json",
        ...     data_root="data/mydataset",
        ...     # The `img_path` field in the train annotation file is relative
        ...     # to the `train` folder.
        ...     data_prefix='train',
        ... )
        >>> train_dataset = build_dataset(train_cfg)

    Or we can put all files in the same folder: ::

        mmclassification/
        └── data
            └── mydataset
                 ├── train.json
                 ├── test.json
                 ├── val.json
                 ├── a.jpg
                 ├── b.jpg
                 ├── c.jpg
                 └── ...

    And we can use the below config to build datasets:

    .. code:: python

        >>> from mmcls.datasets import build_dataset
        >>> train_cfg = dict(
        ...     type="MultiTaskDataset",
        ...     ann_file="train.json",
        ...     data_root="data/mydataset",
        ...     # the `data_prefix` is not required since all paths are
        ...     # relative to the `data_root`.
        ... )
        >>> train_dataset = build_dataset(train_cfg)


    Args:
        ann_file (str): The annotation file path. It can be either absolute
            path or relative path to the ``data_root``.
        metainfo (dict, optional): The extra meta information. It should be
            a dict with the same format as the ``"metainfo"`` field in the
            annotation file. Defaults to None.
        data_root (str, optional): The root path of the data directory. It's
            the prefix of the ``data_prefix`` and the ``ann_file``. And it can
            be a remote path like "s3://openmmlab/xxx/". Defaults to None.
        data_prefix (str, optional): The base folder relative to the
            ``data_root`` for the ``"img_path"`` field in the annotation file.
            Defaults to None.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        test_mode (bool): in train mode or test mode. Defaults to False.
    """
    METAINFO = dict()

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)
        self._metainfo_override = metainfo

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=transforms,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @classmethod
    def _get_meta_info(cls, in_metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            in_metainfo (dict): Meta information dict.

        Returns:
            dict: Parsed meta information.
        """
        # `cls.METAINFO` will be overwritten by in_meta
        metainfo = copy.deepcopy(cls.METAINFO)
        if in_metainfo is None:
            return metainfo

        metainfo.update(in_metainfo)

        return metainfo

    def load_data_list(self):
        """Load annotations from an annotation file.

        Returns:
            list[dict]: A list of annotation.
        """
        annotations = mmengine.load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations:
            raise ValueError('The annotation file must have the `data_list` '
                             'field.')
        metainfo = annotations.get('metainfo', {})
        raw_data_list = annotations['data_list']

        # Set meta information.
        assert isinstance(metainfo, dict), 'The `metainfo` field in the '\
            f'annotation file should be a dict, but got {type(metainfo)}'
        self._metainfo = self._get_meta_info(metainfo)
        if self._metainfo_override is not None:
            self._metainfo.update(self._metainfo_override)

        data_list = []
        for i, raw_data in enumerate(raw_data_list):
            try:
                data_list.append(self.parse_data_info(raw_data))
            except AssertionError as e:
                raise RuntimeError(
                    f'The format check fails during parse the item {i} of '
                    f'the annotation file with error: {e}')
        return data_list

    def parse_data_info(self, raw_data):
        """Parse raw annotation to target format.

        This method will return a dict which contains the data information of a
        sample.

        Args:
            raw_data (dict): Raw data information load from ``ann_file``

        Returns:
            dict: Parsed annotation.
        """
        assert isinstance(raw_data, dict), \
            f'The item should be a dict, but got {type(raw_data)}'
        assert 'img_path' in raw_data, \
            "The item doesn't have `img_path` field."
        img_prefix = self.data_prefix['img_path']
        data = dict(
            img_path=mmengine.join_path(img_prefix, raw_data['img_path']),
            gt_label=raw_data['gt_label'],
        )
        return data

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = [f'Number of samples: \t{self.__len__()}']
        if self.data_root is not None:
            body.append(f'Root location: \t{self.data_root}')
        body.append(f'Annotation file: \t{self.ann_file}')
        if self.data_prefix is not None:
            body.append(f'Prefix of images: \t{self.data_prefix}')
        # -------------------- extra repr --------------------
        tasks = self.metainfo['tasks']
        body.append(f'For {len(tasks)} tasks')
        for task in tasks:
            body.append(f' {task} ')
        # ----------------------------------------------------

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)
