# jax_train.py
# Optimized for TPU v5e-8 on Kaggle
import argparse
import os
import sys
import time
import functools
from typing import Any

# Add current directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import train_state, checkpoints
from flax import jax_utils
from flax.jax_utils import prefetch_to_device

# Import using importlib to ensure we get the correct file
import importlib.util

# Load jax_models with error handling
_models_path = os.path.join(_script_dir, "jax_models.py")
if not os.path.exists(_models_path):
    raise FileNotFoundError(f"jax_models.py not found at {_models_path}")

try:
    spec_models = importlib.util.spec_from_file_location("jax_models", _models_path)
    jax_models = importlib.util.module_from_spec(spec_models)
    spec_models.loader.exec_module(jax_models)
    
    if not hasattr(jax_models, 'JIPNetFullFlax'):
        raise AttributeError(
            f"JIPNetFullFlax not found in jax_models. Available: {[x for x in dir(jax_models) if not x.startswith('_')]}"
        )
    JIPNetFullFlax = jax_models.JIPNetFullFlax
except Exception as e:
    print(f"Error loading jax_models from {_models_path}:")
    import traceback
    traceback.print_exc()
    raise

# Load jax_data
_data_path = os.path.join(_script_dir, "jax_data.py")
if not os.path.exists(_data_path):
    raise FileNotFoundError(f"jax_data.py not found at {_data_path}")

try:
    spec_data = importlib.util.spec_from_file_location("jax_data", _data_path)
    jax_data = importlib.util.module_from_spec(spec_data)
    spec_data.loader.exec_module(jax_data)
    read_pair_list = jax_data.read_pair_list
    pair_generator = jax_data.pair_generator
except Exception as e:
    print(f"Error loading jax_data from {_data_path}:")
    import traceback
    traceback.print_exc()
    raise


# ---------- TPU/Device Setup ----------

def setup_device():
    """Setup JAX for TPU/GPU/CPU with proper configuration."""
    # Print device info
    devices = jax.devices()
    n_devices = len(devices)
    device_kind = devices[0].device_kind if devices else "unknown"
    
    print(f"JAX devices: {n_devices} x {device_kind}")
    print(f"Devices: {devices}")
    
    # Check if TPU
    is_tpu = "TPU" in device_kind.upper()
    if is_tpu:
        print("✓ Running on TPU - using bfloat16 for optimal performance")
    
    return n_devices, is_tpu


# ---------- Losses (with bfloat16 support) ----------

def binary_focal_loss(pred: jnp.ndarray,
                      target: jnp.ndarray,
                      gamma: float = 2.0,
                      alpha: float = 0.25,
                      eps: float = 1e-6) -> jnp.ndarray:
    """
    pred: [B,1] in [0,1]
    target: [B,1] in {0,1}
    """
    # Cast to float32 for loss computation (numerical stability)
    pred = pred.astype(jnp.float32)
    target = target.astype(jnp.float32)
    
    p_t = target * pred + (1.0 - target) * (1.0 - pred)
    focal = -alpha * (1.0 - p_t) ** gamma * jnp.log(p_t + eps)
    return jnp.mean(focal)


def inbatch_ranking_loss(score: jnp.ndarray,
                         target: jnp.ndarray,
                         margin: float = 0.2,
                         key: Any = None) -> jnp.ndarray:
    """
    Random in-batch ranking loss.
    Fixed to avoid boolean indexing issues in JAX.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Cast to float32 for loss computation
    score = score.astype(jnp.float32).squeeze(-1)
    target = target.astype(jnp.float32).squeeze(-1)

    pos_mask = (target == 1.0)
    neg_mask = (target == 0.0)

    n_pos = jnp.sum(pos_mask.astype(jnp.float32))
    n_neg = jnp.sum(neg_mask.astype(jnp.float32))

    # If no positives or negatives, return 0
    has_both = (n_pos > 0) & (n_neg > 0)
    
    # Compute loss for all pairs, then mask
    batch_size = score.shape[0]
    
    # Expand to [batch, batch] for pairwise comparison
    pos_scores_expanded = jnp.expand_dims(score, 0)  # [1, B]
    neg_scores_expanded = jnp.expand_dims(score, 1)  # [B, 1]
    
    # Compute ranking loss for all pairs
    # loss[i,j] = max(0, margin - pos[i] + neg[j])
    # But we only want pairs where i is positive and j is negative
    pairwise_loss = jnp.maximum(0.0, margin - pos_scores_expanded + neg_scores_expanded)
    
    # Mask: only keep pairs where first is positive and second is negative
    valid_mask = jnp.expand_dims(pos_mask.astype(jnp.float32), 0) * \
                 jnp.expand_dims(neg_mask.astype(jnp.float32), 1)
    
    # Sample one negative per positive using random selection
    # For each positive, pick a random negative
    neg_indices = jnp.where(neg_mask, jnp.arange(batch_size), -1)
    pos_indices = jnp.where(pos_mask, jnp.arange(batch_size), -1)
    
    # Sample random negative index for each position
    random_idx = jax.random.randint(key, (batch_size,), minval=0, maxval=batch_size)
    
    # Get the sampled negative scores
    # Use gather with the random indices, but only for valid negatives
    neg_scores_all = jnp.where(neg_mask, score, 0.0)
    sampled_neg_scores = neg_scores_all[random_idx % batch_size]
    
    # Compute loss: margin - pos_score + sampled_neg_score
    # Only for positions that are positive
    loss_per_pos = jnp.maximum(0.0, margin - score + sampled_neg_scores)
    loss_per_pos = loss_per_pos * pos_mask.astype(jnp.float32)
    
    # Average over positives
    total_loss = jnp.sum(loss_per_pos)
    avg_loss = total_loss / jnp.maximum(n_pos, 1.0)
    
    # Return 0 if no valid pairs
    return avg_loss * has_both.astype(jnp.float32)


# ---------- Eval helpers ----------

def tar_at_far(scores, targets, far_list):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    pos = scores[targets == 1.0]
    neg = scores[targets == 0.0]

    neg_sorted = np.sort(neg)[::-1]
    results = []
    for FAR in far_list:
        idx = int(FAR * len(neg_sorted))
        if idx >= len(neg_sorted) or len(neg_sorted) == 0:
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


def evaluate_checkpoint(cfg,
                        ckpt_dir: str,
                        test_npy: str,
                        data_root: str,
                        global_batch: int = 64,
                        far_list=None):
    """Minimal eval pass used at the end of training."""
    far_list = far_list or [1e-1, 1e-2, 1e-3, 1e-4]
    if not test_npy:
        return
    if not os.path.exists(test_npy):
        print(f"[Eval] Skipping evaluation because {test_npy} was not found.")
        return

    print(f"\n[Eval] Running evaluation on {test_npy}")
    n_devices = jax.device_count()
    print("[Eval] Devices:", jax.devices())

    model = JIPNetFullFlax(
        input_size=cfg["model_cfg"]["input_size"],
        global_hidden_dim=cfg["model_cfg"]["global_hidden_dim"],
        transformer_layers=cfg["model_cfg"]["transformer_layers"],
        transformer_heads=cfg["model_cfg"]["transformer_heads"],
    )
    dummy = jnp.zeros(
        (1,
         cfg["model_cfg"]["input_size"],
         cfg["model_cfg"]["input_size"],
         1),
        dtype=jnp.float32
    )
    variables = model.init(jax.random.PRNGKey(0),
                           dummy, dummy, dummy, dummy,
                           cfg["model_cfg"]["fusion_alpha"])
    params = variables["params"]

    restore_path = ckpt_dir
    if not os.path.isdir(restore_path):
        restore_path = os.path.dirname(restore_path)
    print(f"[Eval] Restoring checkpoint from {restore_path}")
    params = checkpoints.restore_checkpoint(restore_path, target=params)
    params = jax_utils.replicate(params)

    @jax.pmap
    def infer(params, batch):
        score = model.apply({"params": params},
                            batch["img1"], batch["img2"],
                            batch["mask1"], batch["mask2"],
                            cfg["model_cfg"]["fusion_alpha"])
        return score

    test_info = read_pair_list(test_npy)
    print(f"[Eval] Loaded {len(test_info)} pairs")

    if global_batch % n_devices != 0:
        global_batch = n_devices * max(1, global_batch // n_devices)
    print(f"[Eval] Global batch {global_batch} (per-device {global_batch // n_devices})")

    gen = pair_generator(
        test_info,
        batch_size=global_batch,
        input_size=cfg["model_cfg"]["input_size"],
        use_augmentation=False,
        use_mask=True,
        shuffle=False,
        data_root=data_root,
    )

    steps = len(test_info) // global_batch + int(len(test_info) % global_batch > 0)
    all_scores, all_targets = [], []

    total_infer_time = 0.0
    total_samples = 0

    for _ in range(steps):
        batch_np = next(gen)
        batch = {k: jnp.array(v) for k, v in batch_np.items()}
        batch = shard_batch(batch, n_devices)
        batch_start = time.time()
        score = infer(params, batch)
        score = np.array(score).reshape(-1)
        batch_time = time.time() - batch_start
        total_infer_time += batch_time
        all_scores.append(score)
        all_targets.append(batch_np["target"].reshape(-1))
        total_samples += batch_np["target"].shape[0]

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    results = tar_at_far(scores, targets, far_list)
    eer, eer_thr = compute_eer(scores, targets)

    print("[Eval] TAR@FAR results:")
    for FAR, thr, TAR in results:
        if thr is None:
            print(f"  FAR={FAR:.1e}: not enough negatives")
        else:
            print(f"  FAR={FAR:.1e}: thr={thr:.4f}, TAR={TAR:.4f}")

    if eer is None:
        print("[Eval] EER: not enough positive/negative samples")
    else:
        print(f"[Eval] EER: {eer*100:.2f}% at thr={eer_thr:.4f}")

    if total_infer_time > 0 and total_samples > 0:
        print(f"[Eval] Inference time: {total_infer_time:.2f}s total "
              f"({total_samples/total_infer_time:.2f} samples/sec)")


# ---------- TrainState ----------

class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, config, is_tpu=False):
    """Create train state with optional bfloat16 for TPU."""
    
    # Determine dtype based on device
    param_dtype = jnp.bfloat16 if is_tpu else jnp.float32
    
    model = JIPNetFullFlax(
        input_size=config["model_cfg"]["input_size"],
        global_hidden_dim=config["model_cfg"]["global_hidden_dim"],
        transformer_layers=config["model_cfg"]["transformer_layers"],
        transformer_heads=config["model_cfg"]["transformer_heads"],
        dtype=param_dtype,
    )
    
    dummy = jnp.zeros(
        (1,
         config["model_cfg"]["input_size"],
         config["model_cfg"]["input_size"],
         1),
        dtype=param_dtype
    )

    variables = model.init(
        rng, dummy, dummy, dummy, dummy,
        fusion_alpha=config["model_cfg"]["fusion_alpha"]
    )
    params = variables["params"]

    # Use learning rate schedule with warmup
    lr = config["train_cfg"]["lr"]
    warmup_steps = config["train_cfg"].get("warmup_steps", 1000)
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=config["train_cfg"]["epochs"] * 1000,  # Approximate
        end_value=lr * 0.01,
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adamw(schedule, weight_decay=0.01),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ---------- Multi-device helpers ----------

def shard_batch(batch, n_devices):
    """[B,...] -> [n_devices, B/n_devices, ...]"""
    def _shard(x):
        B = x.shape[0]
        if B % n_devices != 0:
            # Pad batch to be divisible
            pad_size = n_devices - (B % n_devices)
            x = jnp.concatenate([x, jnp.zeros((pad_size,) + x.shape[1:], dtype=x.dtype)], axis=0)
            B = x.shape[0]
        return x.reshape((n_devices, B // n_devices) + x.shape[1:])
    return {k: _shard(v) for k, v in batch.items()}


def data_iterator_with_prefetch(data_gen, n_devices, prefetch_size=2, is_tpu=False):
    """
    Create a prefetching data iterator for efficient TPU/GPU utilization.
    Converts data to appropriate dtype and shards across devices.
    """
    dtype = jnp.bfloat16 if is_tpu else jnp.float32
    
    def prepare_batch(batch_np):
        batch = {k: jnp.array(v, dtype=dtype) for k, v in batch_np.items()}
        # Keep target as float32 for loss computation
        batch["target"] = jnp.array(batch_np["target"], dtype=jnp.float32)
        return shard_batch(batch, n_devices)
    
    def gen():
        for batch_np in data_gen:
            yield prepare_batch(batch_np)
    
    # Prefetch to device for better overlap
    return prefetch_to_device(gen(), prefetch_size)


def make_train_step(config, is_tpu=False):
    """Create pmapped train step function."""
    
    margin = config["train_cfg"]["ranking_margin"]
    dtype = jnp.bfloat16 if is_tpu else jnp.float32

    def train_step(state: TrainState,
                   batch: dict,
                   fusion_alpha: float,
                   ranking_weight: float,
                   rng: Any):
        """
        Single train step running on each device.
        """

        def loss_fn(params):
            score = state.apply_fn({"params": params},
                                   batch["img1"], batch["img2"],
                                   batch["mask1"], batch["mask2"],
                                   fusion_alpha)

            # Compute losses in float32 for stability
            focal = binary_focal_loss(score, batch["target"])
            rank = inbatch_ranking_loss(score, batch["target"],
                                        margin=margin, key=rng)
            loss = focal + ranking_weight * rank
            return loss, (focal, rank, score)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (focal, rank, score)), grads = grad_fn(state.params)

        # Average grads/metrics over devices
        grads = jax.lax.pmean(grads, axis_name="dev")
        loss = jax.lax.pmean(loss, axis_name="dev")
        focal = jax.lax.pmean(focal, axis_name="dev")
        rank = jax.lax.pmean(rank, axis_name="dev")
        score_mean = jax.lax.pmean(jnp.mean(score.astype(jnp.float32)), axis_name="dev")

        state = state.apply_gradients(grads=grads)

        metrics = {
            "loss": loss,
            "focal": focal,
            "rank": rank,
            "score_mean": score_mean,
        }
        return state, metrics

    # donate_argnums=(0,) tells JAX it can reuse the memory of state
    # static_broadcasted_argnums marks scalar arguments that don't need to be mapped
    return jax.pmap(train_step, axis_name="dev", donate_argnums=(0,), 
                    static_broadcasted_argnums=(2, 3))  # fusion_alpha and ranking_weight are static


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="jax_config_sd300.yaml")
    parser.add_argument("--train_npy", type=str,
                        required=True)
    parser.add_argument("--valid_npy", type=str,
                        required=False, default=None)
    parser.add_argument("--data_root", type=str,
                        required=False, default=None,
                        help="Root directory to prepend to image paths")
    parser.add_argument("--output_dir", type=str,
                        default="./saved_jax_sd300")
    # wandb arguments
    parser.add_argument("--use_wandb", type=int, default=0,
                        help="Whether to use wandb (0 or 1)")
    parser.add_argument("--wandb_project", type=str, default="jipnet-jax",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup devices (TPU/GPU/CPU)
    n_devices, is_tpu = setup_device()

    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            
            # Set API key from environment if available (for Kaggle)
            # On Kaggle, set it via: os.environ['WANDB_API_KEY'] = your_key
            # Or use kaggle_secrets: 
            #   from kaggle_secrets import UserSecretsClient
            #   user_secrets = UserSecretsClient()
            #   os.environ['WANDB_API_KEY'] = user_secrets.get_secret("wandb_api_key")
            
            # Try to login if API key is set, but don't fail if it doesn't work
            if 'WANDB_API_KEY' in os.environ:
                try:
                    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
                except Exception as e:
                    print(f"Warning: wandb.login() failed (this is OK if key is already set): {e}")
            
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={"args": vars(args), "is_tpu": is_tpu, "n_devices": n_devices},
                settings=wandb.Settings(_disable_stats=True)  # Disable stats collection on Kaggle
            )
            print(f"Wandb initialized: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
            args.use_wandb = 0
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}")
            print("Continuing without wandb logging...")
            args.use_wandb = 0
            wandb_run = None

    # Handle config path - if relative, look in jax/ directory
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # Try in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Adjust batch size for TPU (should be divisible by n_devices)
    global_batch = config["train_cfg"]["batch_size"]
    if global_batch % n_devices != 0:
        new_batch = (global_batch // n_devices + 1) * n_devices
        print(f"Adjusting batch_size from {global_batch} to {new_batch} for {n_devices} devices")
        config["train_cfg"]["batch_size"] = new_batch
        global_batch = new_batch

    per_device_batch = global_batch // n_devices
    print(f"Global batch: {global_batch}, Per-device batch: {per_device_batch}")

    # Log config to wandb
    if wandb_run:
        wandb_run.config.update(config)
        wandb_run.config.update({"per_device_batch": per_device_batch})

    os.makedirs(args.output_dir, exist_ok=True)

    # Create train state with TPU optimization
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config, is_tpu=is_tpu)
    state = jax_utils.replicate(state)

    # Load data
    train_info = read_pair_list(args.train_npy)
    print(f"Loaded {len(train_info)} training pairs")

    num_epochs = config["train_cfg"]["epochs"]
    steps_per_epoch = len(train_info) // global_batch

    warmup_epochs = config["train_cfg"]["warmup_epochs"]
    fusion_alpha_warm = config["train_cfg"]["fusion_alpha_warm"]
    hard_start = config["train_cfg"]["hard_negative_start_epoch"]
    fusion_alpha_norm = config["model_cfg"]["fusion_alpha"]
    use_ranking = config["train_cfg"]["use_ranking_loss"]
    wB = config["train_cfg"]["ranking_weight_phaseB"]
    wC = config["train_cfg"]["ranking_weight_phaseC"]

    # Create pmapped train step
    p_train_step = make_train_step(config, is_tpu=is_tpu)

    rng = jax.random.PRNGKey(42)

    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"Using {'bfloat16' if is_tpu else 'float32'} precision")
    print(f"{'='*60}\n")

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Phase schedule
        if epoch < warmup_epochs:
            phase = "A_warmup"
            fusion_alpha = fusion_alpha_warm
            rank_w = 0.0
        elif epoch < hard_start:
            phase = "B_refine"
            fusion_alpha = fusion_alpha_norm
            rank_w = wB if use_ranking else 0.0
        else:
            phase = "C_hardneg"
            fusion_alpha = fusion_alpha_norm
            rank_w = wC if use_ranking else 0.0

        print(f"\nEpoch {epoch}/{num_epochs} phase={phase} "
              f"alpha={fusion_alpha:.2f} rank_w={rank_w:.2f}")

        # Create fresh data generator for each epoch
        train_gen = pair_generator(
            train_info,
            batch_size=global_batch,
            input_size=config["model_cfg"]["input_size"],
            use_augmentation=True,
            use_mask=True,
            shuffle=True,
            data_root=args.data_root,
        )

        epoch_loss = 0.0
        epoch_focal = 0.0
        epoch_rank = 0.0
        step_times = []

        for step in range(steps_per_epoch):
            step_start = time.time()
            
            # Get batch and prepare for devices
            batch_np = next(train_gen)
            dtype = jnp.bfloat16 if is_tpu else jnp.float32
            batch = {k: jnp.array(v, dtype=dtype) for k, v in batch_np.items()}
            batch["target"] = jnp.array(batch_np["target"], dtype=jnp.float32)
            batch = shard_batch(batch, n_devices)

            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, n_devices)

            state, metrics = p_train_step(state, batch,
                                          fusion_alpha, rank_w,
                                          step_keys)

            # Block until computation is done (for accurate timing)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            
            step_time = time.time() - step_start
            step_times.append(step_time)

            # Get metrics from device 0
            m0 = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
            epoch_loss += m0["loss"] / steps_per_epoch
            epoch_focal += m0["focal"] / steps_per_epoch
            epoch_rank += m0["rank"] / steps_per_epoch

            if (step + 1) % 50 == 0:
                avg_step_time = np.mean(step_times[-50:])
                samples_per_sec = global_batch / avg_step_time
                print(f"  step {step+1}/{steps_per_epoch} "
                      f"loss={m0['loss']:.4f} focal={m0['focal']:.4f} "
                      f"rank={m0['rank']:.4f} "
                      f"({samples_per_sec:.1f} samples/sec)")
                
                if wandb_run:
                    wandb_run.log({
                        "step": epoch * steps_per_epoch + step,
                        "step_loss": m0["loss"],
                        "step_focal": m0["focal"],
                        "step_rank": m0["rank"],
                        "step_score_mean": m0["score_mean"],
                        "samples_per_sec": samples_per_sec,
                    })

        epoch_time = time.time() - epoch_start_time
        avg_step_time = np.mean(step_times)
        
        print(f"Epoch {epoch} DONE: loss={epoch_loss:.4f} "
              f"focal={epoch_focal:.4f} rank={epoch_rank:.4f} "
              f"({epoch_time:.1f}s, {global_batch/avg_step_time:.1f} samples/sec avg)")

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "epoch_focal": epoch_focal,
                "epoch_rank": epoch_rank,
                "phase": phase,
                "fusion_alpha": fusion_alpha,
                "rank_weight": rank_w,
                "epoch_time": epoch_time,
            })

        # Save checkpoint
        params_cpu = jax.device_get(jax_utils.unreplicate(state).params)
        checkpoints.save_checkpoint(args.output_dir,
                                    target=params_cpu,
                                    step=epoch,
                                    overwrite=True)

    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Training finished in {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    # Final save
    params_cpu = jax.device_get(jax_utils.unreplicate(state).params)
    checkpoints.save_checkpoint(args.output_dir,
                                target=params_cpu,
                                step=num_epochs,
                                overwrite=True)

    if args.valid_npy:
        final_ckpt_dir = os.path.join(args.output_dir, f"checkpoint_{num_epochs}")
        evaluate_checkpoint(cfg=config,
                            ckpt_dir=final_ckpt_dir,
                            test_npy=args.valid_npy,
                            data_root=args.data_root,
                            global_batch=config["train_cfg"]["batch_size"])

    if wandb_run:
        wandb_run.log({"total_training_time_minutes": total_time / 60})
        wandb_run.finish()


if __name__ == "__main__":
    main()
