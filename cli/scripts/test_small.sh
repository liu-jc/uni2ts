for lr in 1e-4
do
  for n_epoch in 5000
  do
    for mode in M S
    do
      python cli/eval_last_checkpoints_test.py --run_name bs-256-${n_epoch}-lr${lr}-mskrate0.15-0.5 --model_name moirai_small --mode ${mode}
    done
  done
done