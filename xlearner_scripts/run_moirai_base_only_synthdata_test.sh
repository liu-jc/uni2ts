 #!/bin/bash
set -e
cd /export/home/juncheng-liu/uni2ts
. venv/bin/activate
which python
python -m cli.train -cp conf/pretrain data=lotsa_v1_weighted model=moirai_small trainer.max_epochs=5_000 trainer.accumulate_grad_batches=1 run_name=small_500k_0.7_resmlp1_fix16_bf16_r12_top2