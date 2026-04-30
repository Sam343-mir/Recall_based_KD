

python training.py \
  --old_tiles_dir /path/to/old_tiles \
  --new_tiles_dir /path/to/new_tiles \
  --val_tiles_dir /path/to/val_tiles \
  --teacher_model /path/to/teacher_model.h5 \
  --output_dir /path/to/output_run \
  --num_old_tiles 2000 \
  --alpha 0.4 \
  --curriculum 4

        