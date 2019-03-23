#!/usr/bin/env bash

# SEEDS ="0 14 57 1234"

python mr_train.py --seed 0 --data_seed 0 --train /lfs/1/heartmri/mr/train --dev /lfs/1/heartmri/mr/dev --test /lfs/1/heartmri/mr/test --checkpoint_dir mr_checkpoints_seed1 -E 20
python mr_train.py --seed 14 --data_seed 14 --train /lfs/1/heartmri/mr/train --dev /lfs/1/heartmri/mr/dev --test /lfs/1/heartmri/mr/test --checkpoint_dir mr_checkpoints_seed2 -E 20
python mr_train.py --seed 57 --data_seed 57 --train /lfs/1/heartmri/mr/train --dev /lfs/1/heartmri/mr/dev --test /lfs/1/heartmri/mr/test --checkpoint_dir mr_checkpoints_seed3 -E 20
python mr_train.py --seed 1234 --data_seed 1234 --train /lfs/1/heartmri/mr/train --dev /lfs/1/heartmri/mr/dev --test /lfs/1/heartmri/mr/test --checkpoint_dir mr_checkpoints_seed4 -E 20
