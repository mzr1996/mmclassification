import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class DummyImageNet(BaseDataset):
    """`Dummy ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py  # noqa: E501
    """
    dummy_images = {
        i: np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for i in range(1000)
    }

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        if test_mode:
            self.size = 50000
        else:
            self.size = 1281167

        super().__init__(
            data_prefix,
            pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode)

    def load_annotations(self):
        data_infos = []
        for i in range(self.size):
            gt_label = i % 1000
            info = {'img_prefix': self.data_prefix}
            info['img'] = self.dummy_images[gt_label]
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
