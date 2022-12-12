_base_ = ['./convnext/convnext-small_32xb128_in1k.py']

# MMClassification model
base_model = _base_.pop('model')
base_model['init_cfg'] = dict(
    type='Pretrained', checkpoint='./convnext-s_init_converted.pth')
base_model['backbone']['drop_path_rate'] = 0

# Timm model
import sys

sys.path.append('/mm_model/mazerun/Git/ConvNeXt')
custom_imports = dict(imports='models.convnext')

model2 = dict(
    type='TimmClassifier',
    model_name='convnext_small',
    layer_scale_init_value=1e-06,
    head_init_scale=1.0,
    init_cfg=dict(type='Pretrained', checkpoint='./convnext-s_init.pth'),
    loss=base_model['head']['loss'],
)

model = dict(
    type='CompareClassifiers',
    model1=base_model,
    model2=model2,
    train_cfg=base_model.pop('train_cfg', None),
    map_dict=None)

# optimizer settings
base_optimizer = _base_.pop('optim_wrapper')
timm_optimizer = dict(
    constructor='TimmOptimizerConstructor',
    opt='adamw',
    lr=0.004,
    weight_decay=0.05,
    eps=1e-8)
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    model1=base_optimizer,
    model2=timm_optimizer,
)

# common settings
import os

os.environ[
    'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Necessary for deterministic
env_cfg = dict(cudnn_benchmark=False)
randomness = dict(seed=0, deterministic=True)
