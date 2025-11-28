# jax_eval.py
import argparse
import os
import sys

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
                        default="jax_config_sd300.yaml")
    parser.add_argument("--test_npy", type=str,
                        required=True)
    parser.add_argument("--ckpt_dir", type=str,
                        required=True)
    parser.add_argument("--max_batches", type=int,
                        default=None)
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


def main():
    args = parse_args()
    with open(args.config, "r") as f:
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
    params = checkpoints.restore_checkpoint(args.ckpt_dir, target=params)
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

    global_batch = 32
    assert global_batch % n_devices == 0
    gen = pair_generator(test_info, batch_size=global_batch,
                         input_size=input_size,
                         use_augmentation=False,
                         use_mask=True,
                         shuffle=False)

    steps = len(test_info) // global_batch + int(len(test_info) % global_batch > 0)

    all_scores, all_targets = [], []

    for i in range(steps):
        if args.max_batches is not None and i >= args.max_batches:
            break

        batch_np = next(gen)
        batch = {k: jnp.array(v) for k, v in batch_np.items()}
        batch = shard_batch(batch, n_devices)

        score = infer(params, batch)  # [dev, perB, 1]
        score = np.array(score).reshape(-1)

        all_scores.append(score)
        all_targets.append(batch_np["target"].reshape(-1))

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    far_list = [1e-1, 1e-2, 1e-3, 1e-4]
    results = tar_at_far(scores, targets, far_list)

    print("\nTAR@FAR results:")
    for FAR, thr, TAR in results:
        if thr is None:
            print(f"  FAR={FAR:.1e}: not enough negatives")
        else:
            print(f"  FAR={FAR:.1e}: thr={thr:.4f}, TAR={TAR:.4f}")


if __name__ == "__main__":
    main()
