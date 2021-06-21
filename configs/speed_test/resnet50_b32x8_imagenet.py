_base_ = [
    '../_base_/models/resnet50.py', './datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
