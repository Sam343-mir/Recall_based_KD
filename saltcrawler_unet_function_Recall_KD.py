"""
saltcrawler_unet_function_Recall_KD.py

This code contains all the helper functions for Recall-based Knowledge Distillation (KD) in an
incremental segmentation setting.

Expected tile format
--------------------
Each `.npz` file must contain:
- arr_0: input image (H x W) 
- arr_1: target mask (H x W)

"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# This guide can only be run with the TensorFlow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

# ---- Reproducibility helpers -------------------------------------------------

def set_global_determinism(seed: int = 1) -> None:
    """
    Set random seeds and TensorFlow determinism flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Fix randomization in CPU data-loading parallelism (optional but helpful).
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        # In some environments these calls may fail; ignore safely.
        pass


# ---- Data discovery & loading ------------------------------------------------

def discover_npz_tiles(tiles_dir: Path) -> List[Path]:
    """
    Discover `.npz` tiles under a directory.

    Parameters
    ----------
    tiles_dir : Path
        Directory containing tiles.

    Returns
    -------
    List[Path]
        Sorted list of tile paths.
    """
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory does not exist: {tiles_dir}")
    tiles = sorted(tiles_dir.rglob("*.npz"))
    if not tiles:
        raise FileNotFoundError(f"No .npz tiles found under: {tiles_dir}")
    return tiles


def load_tiles(tile_paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load tiles into memory (inputs and targets).

    Parameters
    ----------
    tile_paths : Iterable[Path]
        Paths to `.npz` tiles.

    Returns
    -------
    input_imgs : np.ndarray
        Shape: (N, H, W, 1) depending on tile contents after transpose.
    targets : np.ndarray
        Shape: (N, H, W, 1) depending on tile contents after transpose.
    """
    tile_paths = list(tile_paths)
    if len(tile_paths) == 0:
        raise ValueError("No tile paths provided to load_tiles().")

    # Peek first tile for shape
    first = np.load(tile_paths[0])
    x0 = first["arr_0"].transpose().astype("float32")
    y0 = first["arr_1"].transpose().astype("uint8")
    tile_size = x0.shape

    input_imgs = np.zeros((len(tile_paths),) + tile_size, dtype="float32")
    targets = np.zeros((len(tile_paths),) + tile_size, dtype="uint8")

    input_imgs[0] = x0
    targets[0] = y0

    for i in range(1, len(tile_paths)):
        data = np.load(tile_paths[i])
        input_imgs[i] = data["arr_0"].transpose().astype("float32")
        targets[i] = data["arr_1"].transpose().astype("uint8")

    return input_imgs, targets


# ---- Losses ------------------------------------------------------------------

def binary_kd_kl_loss_with_temp(teacher_probs: tf.Tensor,
                                student_probs: tf.Tensor,
                                temperature: float = 2.0,
                                eps: float = 1e-7) -> tf.Tensor:

    """
    Temperature-scaled KD for Bernoulli outputs when inputs are probabilities.
    """

    t = tf.clip_by_value(teacher_probs, eps, 1.0 - eps)
    s = tf.clip_by_value(student_probs, eps, 1.0 - eps)
 
    # probs -> logits
    t_logit = tf.math.log(t) - tf.math.log(1.0 - t)
    s_logit = tf.math.log(s) - tf.math.log(1.0 - s)
 
    # apply temperature in logit space
    t_soft = tf.nn.sigmoid(t_logit / temperature)
    s_soft = tf.nn.sigmoid(s_logit / temperature)
 
    # KL(Teacher || Student) with softened probs
    kl = t_soft * tf.math.log(t_soft / s_soft) + (1.0 - t_soft) * tf.math.log((1.0 - t_soft) / (1.0 - s_soft))
 
    # Common KD scaling
    return tf.reduce_mean(kl) * (temperature ** 2)


# ---- Training ----------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training."""
    epochs: int = 30
    batch_size: int = 16
    alpha: float = 0.4
    curriculum: int = 4
    num_old_tiles_to_select: int = 2000
    seed: int = 1
    save_every: int = 5
    learning_rate: float = 1e-3


def _build_train_generator(old_x: np.ndarray, old_y: np.ndarray,
                           new_x: np.ndarray, new_y: np.ndarray,
                           batch_size: int,
                           curriculum: int,
                           seed: int) -> Iterable:
    """
    Build a training generator matching the original curriculum behavior.

    Each item yields (data, target, binary_labels) where binary_labels==1 for old tiles
    (KD enabled) and 0 for new tiles (KD disabled). This is done to keep track of whether to apply distillaion loss on a certain batch or not.
    """
    binary_old = np.ones_like(old_y)
    binary_new = np.zeros_like(new_y)

    old_ds = tf.data.Dataset.from_tensor_slices((old_x, old_y, binary_old)).shuffle(1024, seed=seed).batch(batch_size)
    new_ds = tf.data.Dataset.from_tensor_slices((new_x, new_y, binary_new)).shuffle(1024, seed=seed).batch(batch_size)

    if curriculum == 2:
        return old_ds.concatenate(new_ds)
    if curriculum == 3:
        return new_ds.concatenate(old_ds)
    if curriculum == 4:
        # Interleave batches: new, old, new, old ... until one dataset ends.
        return [b_new for pair in zip(new_ds, old_ds) for b_new in pair]
    # curriculum == 5
    return old_ds.concatenate(new_ds).shuffle(32, seed=seed)


def train_model(
    old_tiles_dir: Path,
    new_tiles_dir: Path,
    val_tiles_dir: Path,
    teacher_model_path: Path,
    output_dir: Path,
    num_old_tiles_to_select: int = 2000,
    alpha: float = 0.4,
    curriculum: int = 4,
    seed: int = 1,
    epochs: int = 30,
    batch_size: int = 16,
    save_every: int = 5,
) -> Path:
    """
    Train an incremental segmentation model using Recall-based KD.

    Parameters
    ----------
    old_tiles_dir, new_tiles_dir, val_tiles_dir : Path
        Directories containing `.npz` tiles for old training, new training, and validation.
    teacher_model_path : Path
        Path to the pre-trained model used as teacher (and as the initialization for the student).
    output_dir : Path
        Directory where checkpoints and the final model will be saved.
    num_old_tiles_to_select : int
        Number of old tiles to sample for replay.
    alpha : float
        Weight of KD loss on old tiles.
    curriculum : int
        Curriculum strategy (2, 3, 4, or 5).
    seed : int
        Random seed for sampling/shuffling.
    epochs, batch_size, save_every : int
        Training loop settings.

    Returns
    -------
    Path
        Path to the final saved model.
    """
    set_global_determinism(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tiles
    old_tiles_all = discover_npz_tiles(old_tiles_dir)
    new_tiles_all = discover_npz_tiles(new_tiles_dir)
    val_tiles = discover_npz_tiles(val_tiles_dir)

    # Sample old tiles
    rng = random.Random(seed)
    rng.shuffle(old_tiles_all)
    old_tiles = old_tiles_all[: min(num_old_tiles_to_select, len(old_tiles_all))]

    print(f"Discovered tiles:"
          f"\n  old: {len(old_tiles_all)} (using {len(old_tiles)})"
          f"\n  new: {len(new_tiles_all)}"
          f"\n  val: {len(val_tiles)}\n")

    # Load into memory
    old_x, old_y = load_tiles(old_tiles)
    new_x, new_y = load_tiles(new_tiles_all)
    val_x, val_y = load_tiles(val_tiles)

    # Model setup
    teacher_model = load_model(str(teacher_model_path))
    student_model = load_model(str(teacher_model_path))

    if student_model is None:
        raise ValueError("Loaded student model is None. Check the teacher_model_path.")

    opt = keras.optimizers.Adam(learning_rate=1e-3)

    # Because the model output already uses sigmoid in the architecture, from_logits=False.
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_acc = keras.metrics.BinaryAccuracy()
    val_acc = keras.metrics.BinaryAccuracy()

    # Generator
    train_gen = _build_train_generator(old_x, old_y, new_x, new_y, batch_size, curriculum, seed)

    # Training loop (custom) to mimic the original logic
    print("Beginning training...\n")
    for epoch in range(epochs):
        train_losses = []
        val_losses = []

        print(f"Epoch {epoch + 1}/{epochs}")

        for step, (data, target, binary_labels) in enumerate(train_gen):
            with tf.GradientTape() as tape:
                y_s = student_model(data, training=True)

                # Determine if this batch is old (KD enabled) by checking binary_labels.
                # binary_labels has the same shape as target. If any element is 1, it's an old batch.
                is_old_batch = tf.reduce_any(tf.equal(binary_labels, 1))

                def old_loss():
                    y_t = teacher_model(data, training=False)
                    loss_ce = bce(target, tf.squeeze(y_s, axis=-1))
                    loss_kd = binary_kd_kl_loss(y_t, y_s, temperature=2.0)
                    return ((1.0 - alpha) * loss_ce) + (alpha * loss_kd)

                def new_loss():
                    return bce(target, tf.squeeze(y_s, axis=-1))

                loss = tf.cond(is_old_batch, old_loss, new_loss)

            grads = tape.gradient(loss, student_model.trainable_weights)
            opt.apply_gradients(zip(grads, student_model.trainable_weights))

            train_acc.update_state(target, y_s)
            train_losses.append(float(loss.numpy()))

        # Validation
        for step in range(0, len(val_x), batch_size):
            batch_x = val_x[step: step + batch_size]
            batch_y = val_y[step: step + batch_size]
            y_val = student_model(batch_x, training=False)
            vloss = bce(batch_y, tf.squeeze(y_val, axis=-1))
            val_losses.append(float(vloss.numpy()))
            val_acc.update_state(batch_y, y_val)

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        tr_acc = float(train_acc.result().numpy())
        va_acc = float(val_acc.result().numpy())
        train_acc.reset_state()
        val_acc.reset_state()

        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} "
              f"| Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        # Save periodic checkpoints
        if save_every and ((epoch + 1) % save_every == 0):
            ckpt_path = output_dir / f"unet_KD_epoch{epoch+1}.keras"
            student_model.save(str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = output_dir / "unet_saltcrawler_finalModel.keras"
    student_model.save(str(final_path))
    print(f"\nFinal model saved: {final_path}")
    return final_path
