for lr in 1e-3 5e-4
do
  for n_epoch in 5000
  do
    python cli/eval_checkpoints.py --run_name bs-256-${n_epoch}-lr${lr}-mskrate0.15-0.5 --model_name moirai_small --max_epochs ${n_epoch}
  done
done