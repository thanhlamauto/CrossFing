# jax_eval.py
import argparse
import os
import sys
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.training import checkpoints
from flax import jax_utils

from jax_models import JIPNetFullFlax
from jax_data import read_pair_list, pair_generator


def shard_batch(batch, n_devices):
    """[B,...] -> [n_devices, B/n_devices, ...]"""
    def _shard(x):
        B = x.shape[0]
        assert B % n_devices == 0
        return x.reshape((n_devices, B // n_devices) + x.shape[1:])
    return {k: _shard(v) for k, v in batch.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="jax_config_sd300.yaml",
                        help="YAML config file (relative paths are resolved against jax/)")
    parser.add_argument("--test_npy", type=str,
                        required=True,
                        help="Path to npy file that contains info_lst for evaluation")
    parser.add_argument("--data_root", type=str,
                        default=None,
                        help="Optional root dir to prepend to relative image paths")
    parser.add_argument("--ckpt_dir", type=str,
                        required=True,
                        help="Directory that contains Flax checkpoints (e.g. saved_jax_fvc)")
    parser.add_argument("--ckpt_step", type=int,
                        default=None,
                        help="Specific checkpoint step to restore (e.g. 40). "
                             "If None, the latest checkpoint will be used.")
    parser.add_argument("--global_batch", type=int,
                        default=64,
                        help="Global batch size for evaluation (must be divisible by #devices)")
    parser.add_argument("--max_batches", type=int,
                        default=None,
                        help="Optional limit on number of batches to evaluate")
    parser.add_argument("--far_list", type=float, nargs="+",
                        default=[1e-1, 1e-2, 1e-3, 1e-4],
                        help="FAR values to report TAR for")
    return parser.parse_args()


def tar_at_far(scores, targets, far_list=(1e-1, 1e-2, 1e-3, 1e-4)):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    pos = scores[targets == 1.0]
    neg = scores[targets == 0.0]

    neg_sorted = np.sort(neg)[::-1]
    results = []
    for FAR in far_list:
        idx = int(FAR * len(neg_sorted))
        if idx >= len(neg_sorted):
            results.append((FAR, None, None))
            continue
        thr = neg_sorted[idx]
        TAR = np.mean(pos >= thr) if len(pos) > 0 else None
        results.append((FAR, thr, TAR))
    return results


def compute_eer(scores, targets):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    pos = scores[targets == 1.0]
    neg = scores[targets == 0.0]

    if len(pos) == 0 or len(neg) == 0:
        return None, None

    thresholds = np.sort(np.unique(scores))
    best_eer = None
    best_thr = None
    min_diff = float("inf")

    for thr in thresholds:
        far = np.mean(neg >= thr)
        frr = np.mean(pos < thr)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_eer = 0.5 * (far + frr)
            best_thr = thr

    return best_eer, best_thr


def main():
    args = parse_args()

    # Resolve config path relative to this script if needed
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, config_path)
        if os.path.exists(alt):
            config_path = alt

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    n_devices = jax.device_count()
    print("Eval on devices:", n_devices)

    input_size = cfg["model_cfg"]["input_size"]
    fusion_alpha = cfg["model_cfg"]["fusion_alpha"]

    model = JIPNetFullFlax(
        input_size=input_size,
        global_hidden_dim=cfg["model_cfg"]["global_hidden_dim"],
        transformer_layers=cfg["model_cfg"]["transformer_layers"],
        transformer_heads=cfg["model_cfg"]["transformer_heads"],
    )
    dummy = jnp.zeros((1, input_size, input_size, 1), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0),
                           dummy, dummy, dummy, dummy,
                           fusion_alpha=fusion_alpha)
    params = variables["params"]

    # restore ckpt
    ckpt_dir = args.ckpt_dir
    if args.ckpt_step is not None:
        ckpt_dir = os.path.join(ckpt_dir, f"checkpoint_{args.ckpt_step}")
    print(f"Restoring checkpoint from: {ckpt_dir}")
    params = checkpoints.restore_checkpoint(ckpt_dir, target=params)
    params = jax_utils.replicate(params)  # replicate for pmap

    # pmap infer
    @jax.pmap
    def infer(params, batch):
        score = model.apply({"params": params},
                            batch["img1"], batch["img2"],
                            batch["mask1"], batch["mask2"],
                            fusion_alpha)
        return score

    test_info = read_pair_list(args.test_npy)
    print(f"Loaded {len(test_info)} eval pairs from {args.test_npy}")

    global_batch = args.global_batch
    if global_batch % n_devices != 0:
        new_batch = n_devices * max(1, global_batch // n_devices)
        print(f"Adjusting global_batch from {global_batch} to {new_batch} "
              f"to be divisible by {n_devices} devices")
        global_batch = new_batch

    print(f"Global batch: {global_batch}  (per-device: {global_batch // n_devices})")

    gen = pair_generator(test_info, batch_size=global_batch,
                         input_size=input_size,
                         use_augmentation=False,
                         use_mask=True,
                         shuffle=False,
                         data_root=args.data_root)

    steps = len(test_info) // global_batch + int(len(test_info) % global_batch > 0)

    all_scores, all_targets = [], []

    total_infer_time = 0.0
    total_samples = 0

    for i in range(steps):
        if args.max_batches is not None and i >= args.max_batches:
            break

        batch_np = next(gen)
        batch = {k: jnp.array(v) for k, v in batch_np.items()}
        batch = shard_batch(batch, n_devices)

        batch_start = time.time()
        score = infer(params, batch)  # [dev, perB, 1]
        score = np.array(score).reshape(-1)
        batch_time = time.time() - batch_start
        total_infer_time += batch_time

        all_scores.append(score)
        all_targets.append(batch_np["target"].reshape(-1))
        total_samples += batch_np["target"].shape[0]

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    results = tar_at_far(scores, targets, args.far_list)
    eer, eer_thr = compute_eer(scores, targets)

    print("\nTAR@FAR results:")
    for FAR, thr, TAR in results:
        if thr is None:
            print(f"  FAR={FAR:.1e}: not enough negatives")
        else:
            print(f"  FAR={FAR:.1e}: thr={thr:.4f}, TAR={TAR:.4f}")

    if eer is None:
        print("EER: not enough positive/negative samples")
    else:
        print(f"EER: {eer*100:.2f}% at thr={eer_thr:.4f}")

    if total_infer_time > 0 and total_samples > 0:
        print(f"Inference time: {total_infer_time:.2f}s total "
              f"({total_samples/total_infer_time:.2f} samples/sec)")


if __name__ == "__main__":
    main()
