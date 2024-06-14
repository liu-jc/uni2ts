for lr in 1e-4
do
  for n_epoch in 1000 5000
  do
    python cli/eval_last_checkpoints_val_hyperparam.py --run_name bs-256-${n_epoch}-lr${lr}-mskrate0.15-0.5 --model_name moirai_small
  done
done