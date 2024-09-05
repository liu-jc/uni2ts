max_mask_ratio=0.5
min_mask_ratio=0.15
max_epoch=1000
batch_size=256
for max_epoch in 10000
do
  for lr in 1e-3
  do
    echo "python -m cli.train --config-name loss_per_ds run_name=bs-${batch_size}-${max_epoch}-lr(${lr})-mskrate(${min_mask_ratio}-${max_mask_ratio}) model=moirai_small data=lotsa_v1_univar \
    val_data=lsf_val model.lr=${lr}  trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} compile=default"
    python -m cli.train -cp conf/pretrain --config-name moirai_univar run_name=bs-${batch_size}-${max_epoch}-lr${lr}-mskrate${min_mask_ratio}-${max_mask_ratio} model=moirai_base data=lotsa_v1_univar \
    model.lr=${lr} trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} compile=false \
    model.log_loss_per_ds=False
  done
done