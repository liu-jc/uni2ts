#!/bin/bash
set -e
cd /export/home/juncheng-liu/uni2ts
pwd
pip install virtualenv
virtualenv venv
. venv-xlearner/bin/activate
pip install -e '.[notebook]'
wandb login --relogin --host=https://salesforceairesearch.wandb.io local-cad323e35fe33a9878acb22ab41b399081de8a0e
#sh cli/scripts/only_synthdata_scripts/moirai_only_tsmixup_base.sh

max_mask_ratio=0.5
min_mask_ratio=0.15
batch_size=256
precision=bf16
lr=1e-3
max_epoch=10000
python -m cli.train -cp conf/pretrain --config-name moirai_univar_synthdata run_name=bs-${batch_size}-${max_epoch}-lr${lr}-${precision}-mskrate${min_mask_ratio}-${max_mask_ratio} model=moirai_base \
model.lr=${lr} trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} compile=default \
model.log_loss_per_ds=True  data=lotsa_v1_univar_w_synthdata trainer.precision=${precision}