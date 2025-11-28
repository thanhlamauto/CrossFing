# jax_train.py
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import train_state, checkpoints
from flax import jax_utils

from jax_models import JIPNetFullFlax
from jax_data import read_pair_list, pair_generator


# ---------- Losses ----------

def binary_focal_loss(pred: jnp.ndarray,
                      target: jnp.ndarray,
                      gamma: float = 2.0,
                      alpha: float = 0.25,
                      eps: float = 1e-6) -> jnp.ndarray:
    """
    pred: [B,1] in [0,1]
    target: [B,1] in {0,1}
    """
    p_t = target * pred + (1.0 - target) * (1.0 - pred)
    focal = -alpha * (1.0 - p_t) ** gamma * jnp.log(p_t + eps)
    return jnp.mean(focal)


def inbatch_ranking_loss(score: jnp.ndarray,
                         target: jnp.ndarray,
                         margin: float = 0.2,
                         key: jax.random.KeyArray = None) -> jnp.ndarray:
    """
    Random in-batch ranking loss:
      mỗi positive so với 1 negative random trong batch (per-device batch).
    score: [B,1], target: [B,1]
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    score = score.squeeze(-1)
    target = target.squeeze(-1)

    pos_mask = (target == 1.0)
    neg_mask = (target == 0.0)

    n_pos = jnp.sum(pos_mask)
    n_neg = jnp.sum(neg_mask)

    def _no_pos_neg():
        return jnp.array(0.0, dtype=jnp.float32)

    def _compute():
        pos_scores = score[pos_mask]  # [Np]
        neg_scores = score[neg_mask]  # [Nn]
        idx = jax.random.randint(key, (pos_scores.shape[0],),
                                 minval=0, maxval=neg_scores.shape[0])
        neg_sampled = neg_scores[idx]
        loss = jnp.maximum(0.0, margin - pos_scores + neg_sampled)
        return jnp.mean(loss)

    return jax.lax.cond((n_pos > 0) & (n_neg > 0),
                        _compute, _no_pos_neg)


# ---------- TrainState ----------

class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, config):
    model = JIPNetFullFlax(
        input_size=config["model_cfg"]["input_size"],
        global_hidden_dim=config["model_cfg"]["global_hidden_dim"],
        transformer_layers=config["model_cfg"]["transformer_layers"],
        transformer_heads=config["model_cfg"]["transformer_heads"],
    )
    dummy = jnp.zeros(
        (1,
         config["model_cfg"]["input_size"],
         config["model_cfg"]["input_size"],
         1),
        dtype=jnp.float32
    )

    variables = model.init(
        rng, dummy, dummy, dummy, dummy,
        fusion_alpha=config["model_cfg"]["fusion_alpha"]
    )
    params = variables["params"]

    tx = optax.adamw(config["train_cfg"]["lr"])

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
        assert B % n_devices == 0, \
            f"Global batch {B} must be divisible by n_devices {n_devices}"
        return x.reshape((n_devices, B // n_devices) + x.shape[1:])
    return {k: _shard(v) for k, v in batch.items()}


def make_train_step(config):

    margin = config["train_cfg"]["ranking_margin"]

    def train_step(state: TrainState,
                   batch: dict,
                   fusion_alpha: float,
                   ranking_weight: float,
                   rng: jax.random.KeyArray):
        """
        train_step chạy trên 1 device (per-device batch).
        pmap sẽ wrap ngoài.
        """

        def loss_fn(params):
            score = state.apply_fn({"params": params},
                                   batch["img1"], batch["img2"],
                                   batch["mask1"], batch["mask2"],
                                   fusion_alpha)

            focal = binary_focal_loss(score, batch["target"])
            rank = inbatch_ranking_loss(score, batch["target"],
                                        margin=margin, key=rng)
            loss = focal + ranking_weight * rank
            return loss, (focal, rank, score)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (focal, rank, score)), grads = grad_fn(state.params)

        # average grads/metrics over devices
        grads = jax.lax.pmean(grads, axis_name="dev")
        loss = jax.lax.pmean(loss, axis_name="dev")
        focal = jax.lax.pmean(focal, axis_name="dev")
        rank = jax.lax.pmean(rank, axis_name="dev")
        score_mean = jax.lax.pmean(jnp.mean(score), axis_name="dev")

        state = state.apply_gradients(grads=grads)

        metrics = {
            "loss": loss,
            "focal": focal,
            "rank": rank,
            "score_mean": score_mean,
        }
        return state, metrics

    return jax.pmap(train_step, axis_name="dev")  # ✅ multi-device


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

    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={"args": vars(args)}
            )
            print(f"Wandb initialized: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
            args.use_wandb = 0

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Log config to wandb
    if wandb_run:
        wandb_run.config.update(config)

    os.makedirs(args.output_dir, exist_ok=True)

    n_devices = jax.device_count()
    print("JAX devices:", n_devices, jax.devices())

    # init state, then replicate to all devices
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)
    state = jax_utils.replicate(state)  # ✅ replicate params/opt state

    # data
    train_info = read_pair_list(args.train_npy)
    print(f"Loaded {len(train_info)} training pairs")

    global_batch = config["train_cfg"]["batch_size"]
    assert global_batch % n_devices == 0, \
        f"batch_size {global_batch} must be divisible by {n_devices}"

    train_gen = pair_generator(
        train_info,
        batch_size=global_batch,
        input_size=config["model_cfg"]["input_size"],
        use_augmentation=True,
        use_mask=True,
        shuffle=True,
        data_root=args.data_root,
    )

    num_epochs = config["train_cfg"]["epochs"]
    steps_per_epoch = len(train_info) // global_batch

    warmup_epochs = config["train_cfg"]["warmup_epochs"]
    fusion_alpha_warm = config["train_cfg"]["fusion_alpha_warm"]
    hard_start = config["train_cfg"]["hard_negative_start_epoch"]
    fusion_alpha_norm = config["model_cfg"]["fusion_alpha"]
    use_ranking = config["train_cfg"]["use_ranking_loss"]
    wB = config["train_cfg"]["ranking_weight_phaseB"]
    wC = config["train_cfg"]["ranking_weight_phaseC"]

    p_train_step = make_train_step(config)

    rng = jax.random.PRNGKey(42)

    print(f"\nStarting training for {num_epochs} epochs, {steps_per_epoch} steps/epoch")

    for epoch in range(num_epochs):
        # phase schedule
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

        epoch_loss = 0.0
        epoch_focal = 0.0
        epoch_rank = 0.0

        for step in range(steps_per_epoch):
            batch_np = next(train_gen)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}
            batch = shard_batch(batch, n_devices)  # ✅ shard for pmap

            rng, step_key = jax.random.split(rng)
            # per-device rng: [n_devices, 2]
            step_keys = jax.random.split(step_key, n_devices)

            state, metrics = p_train_step(state, batch,
                                          fusion_alpha, rank_w,
                                          step_keys)

            # metrics is replicated; take from device 0
            m0 = jax.tree_util.tree_map(lambda x: x[0], metrics)
            epoch_loss += float(m0["loss"]) / steps_per_epoch
            epoch_focal += float(m0["focal"]) / steps_per_epoch
            epoch_rank += float(m0["rank"]) / steps_per_epoch

            if (step + 1) % 50 == 0:
                print(f"  step {step+1}/{steps_per_epoch} "
                      f"loss={m0['loss']:.4f} focal={m0['focal']:.4f} "
                      f"rank={m0['rank']:.4f}")
                
                # Log step metrics to wandb
                if wandb_run:
                    wandb_run.log({
                        "step": epoch * steps_per_epoch + step,
                        "step_loss": float(m0["loss"]),
                        "step_focal": float(m0["focal"]),
                        "step_rank": float(m0["rank"]),
                        "step_score_mean": float(m0["score_mean"]),
                    })

        print(f"Epoch {epoch} DONE: loss={epoch_loss:.4f} "
              f"focal={epoch_focal:.4f} rank={epoch_rank:.4f}")

        # Log epoch metrics to wandb
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "epoch_focal": epoch_focal,
                "epoch_rank": epoch_rank,
                "phase": phase,
                "fusion_alpha": fusion_alpha,
                "rank_weight": rank_w,
            })

        # save checkpoint (unreplicate first)
        params_cpu = jax.device_get(jax_utils.unreplicate(state).params)
        checkpoints.save_checkpoint(args.output_dir,
                                    target=params_cpu,
                                    step=epoch,
                                    overwrite=True)

    print("Training finished.")

    # final save
    params_cpu = jax.device_get(jax_utils.unreplicate(state).params)
    checkpoints.save_checkpoint(args.output_dir,
                                target=params_cpu,
                                step=num_epochs,
                                overwrite=True)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
