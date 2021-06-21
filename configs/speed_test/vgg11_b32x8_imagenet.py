_base_ = [
    '../_base_/models/vgg11.py',
    './datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]
optimizer = dict(lr=0.01)
