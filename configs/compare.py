_base_ = ['./deit3/deit3-base-p16_64xb64_in1k.py']

# MMClassification model
base_model = _base_.pop('model')
base_model['init_cfg'] = dict(type='Pretrained', checkpoint='./deit3-b_init_converted.pth')
base_model['backbone']['drop_path_rate'] = 0

# Timm model
custom_imports = dict(imports='models_v2')
import sys
sys.path.append("/nvme/mazerun/Git/deit")

model2 = dict(
    type='TimmClassifier',
    model_name='deit_base_patch16_LS',
    img_size=192,
    #  init_cfg=dict(type='Pretrained', checkpoint='./deit3-b_init.pth'),
    loss=base_model['head']['loss'],
)

model = dict(
    type='CompareClassifiers',
    model1=base_model,
    model2=model2,
    train_cfg=base_model.pop('train_cfg', None))

# optimizer settings
base_optimizer = _base_.pop('optim_wrapper')
timm_optimizer = dict(
    constructor='TimmOptimizerConstructor',
    opt='lamb',
    lr=0.003,
    weight_decay=0.05,
    eps=1e-8)
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    model1=base_optimizer,
    model2=timm_optimizer,
)

# common settings
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Necessary for deterministic
env_cfg = dict(cudnn_benchmark=False)
randomness = dict(seed=0, deterministic=True)
