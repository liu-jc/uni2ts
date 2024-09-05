max_mask_ratio=0.5
min_mask_ratio=0.15
max_epoch=1000
batch_size=256
accumulate_factor=1
for max_epoch in 10000
do
  for lr in 1e-3
  do
    echo "python -m cli.train -cp conf/pretrain --config-name loss_per_ds run_name=bs-${batch_size}-accumulate${accumulate_factor}-${max_epoch}-lr${lr}-mskrate${min_mask_ratio}-${max_mask_ratio} model=moirai_base data=lotsa_v1_weighted \
    model.lr=${lr} trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} \
    trainer.accumulate_grad_batches=${accumulate_factor} compile=default model.log_loss_per_ds=True"
    python -m cli.train -cp conf/pretrain --config-name loss_per_ds run_name=bs-${batch_size}-accumulate${accumulate_factor}-${max_epoch}-lr${lr}-mskrate${min_mask_ratio}-${max_mask_ratio} model=moirai_base data=lotsa_v1_weighted \
    model.lr=${lr} trainer.max_epochs=${max_epoch} model.min_mask_ratio=${min_mask_ratio} model.max_mask_ratio=${max_mask_ratio} train_dataloader.batch_size=${batch_size} \
    trainer.accumulate_grad_batches=${accumulate_factor} compile=default model.log_loss_per_ds=True
  done
done

