"""
training.py

Entry point for training an incremental U-Net segmentation model using a
Recall-based Knowledge Distillation (KD) strategy.

Typical usage:
    python training.py \
        --old_tiles_dir /path/to/old/tiles \
        --new_tiles_dir /path/to/new/tiles \
        --val_tiles_dir /path/to/val/tiles \
        --teacher_model /path/to/teacher_or_base_model.h5 \
        --output_dir /path/to/output_runs/run1 \
        --num_old_tiles 2000 \
        --alpha 0.4 \
        --curriculum 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from saltcrawler_unet_function_Recall_KD_v3 import train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    p = argparse.ArgumentParser(description="Recall-based KD incremental training (directory-driven).")

    p.add_argument("--old_tiles_dir", type=str, required=True,
                   help="Directory containing OLD training tiles as .npz files.")
    p.add_argument("--new_tiles_dir", type=str, required=True,
                   help="Directory containing NEW training tiles as .npz files.")
    p.add_argument("--val_tiles_dir", type=str, required=True,
                   help="Directory containing VALIDATION tiles as .npz files.")

    p.add_argument("--teacher_model", type=str, required=True,
                   help="Path to the pre-trained teacher/base model (.h5). Used for KD on old tiles.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for checkpoints and the final trained model.")

    p.add_argument("--num_old_tiles", type=int, default=2000,
                   help="How many OLD tiles to replay (randomly sampled). Default: 2000.")
    p.add_argument("--alpha", type=float, default=0.4,
                   help="Weight of KD loss on old tiles. 0 => no KD, 1 => only KD. Default: 0.4.")
    p.add_argument("--curriculum", type=int, default=4, choices=[2, 3, 4, 5],
                   help=("Batch curriculum strategy: "
                         "2=old then new, 3=new then old, 4=interleaved, 5=concatenate+shuffle. Default: 4"))
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for sampling/shuffling. Default: 1.")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of epochs. Default: 30.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size. Default: 16.")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save a checkpoint every N epochs. Default: 5.")
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    start_time = time.time()
    model_path = train_model(
        old_tiles_dir=Path(args.old_tiles_dir),
        new_tiles_dir=Path(args.new_tiles_dir),
        val_tiles_dir=Path(args.val_tiles_dir),
        teacher_model_path=Path(args.teacher_model),
        output_dir=Path(args.output_dir),
        num_old_tiles_to_select=args.num_old_tiles,
        alpha=args.alpha,
        curriculum=args.curriculum,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )

    print(f"\nTraining complete.\nFinal model saved at: {model_path}")
    print("Time taken to train model --- %.2f seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()