_base_ = [
    '../_base_/models/resnext50_32x4d.py', './datasets/imagenet_bs32',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
