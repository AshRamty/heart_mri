#python mr_framewise_train.py --train /dfs/scratch0/ashwinir/heart_mri/data/mr_framewise_3/train/ --dev /dfs/scratch0/ashwinir/heart_mri/data/mr_framewise_3/dev/  --test /dfs/scratch0/ashwinir/heart_mri/data/mr_framewise_3/test/  --mr_result_filename ./results/first_run --num_workers 16 --batch_size 12 --n_epochs 5

python mr_framewise_train_metal.py --train /dfs/scratch0/ashwinir/heart_mri/data/mr_framewise_3/train/ --dev /dfs/scratch0/ashwinir/heart_mri/data/mr_framewise_3/dev/  --test /dfs/scratch0/ashwinir/heart_mri/data/mr_framewise_3/test/  --mr_result_filename ./results/first_run --config em_config