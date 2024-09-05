max_mask_ratio=0.5
min_mask_ratio=0.15
batch_size=256
precision=bf16
for max_epoch in 10000
do
  for lr in 1e-3
  do
    python -m cli.train -cp conf/pretrain --config-name moirai_only_tsmixup run_name=bs-${batch_size}-${max_epoch}-lr${lr}-${precision}-mskrate${min_mask_ratio}-${max_mask_ratio} model=moirai_base \
    model.lr=${lr} trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} compile=default \
    model.log_loss_per_ds=True  data=tsmixup_10m trainer.precision=${precision}
  done
done