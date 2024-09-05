#!/bin/bash
set -e
cd /export/home/juncheng-liu/uni2ts
pip3 install -e '.[notebook]'
which python
which pip
pip3 install wandb
pip3 install ipdb
wandb login --relogin --host=https://salesforceairesearch.wandb.io local-cad323e35fe33a9878acb22ab41b399081de8a0e
sh cli/scripts/only_synthdata_scripts/moirai_only_synthdata_small.sh