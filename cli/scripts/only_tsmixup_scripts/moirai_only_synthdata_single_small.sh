max_mask_ratio=0.5
min_mask_ratio=0.15
max_epoch=1000
batch_size=256
precision=bf16
for lr in 1e-3
do
  python -m cli.train -cp conf/pretrain --config-name moirai_only_synthdata run_name=bs-${batch_size}-${max_epoch}-lr${lr}-mskrate${min_mask_ratio}-${max_mask_ratio} model=moirai_small \
  model.lr=${lr} trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} compile=default \
  model.log_loss_per_ds=True  data=only_synthdata_single trainer.precision=${precision}
done
