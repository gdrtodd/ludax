# This script is based on the PGX AlphaZero training script, available at:
# https://github.com/sotetsuk/pgx/blob/18799f81a03651e7de8fb9dc79daee9090e2e695/examples/alphazero/train.py
# We have modified it to work with the LDX environment and added some additional features.
# The original script was licensed under the Apache License, Version 2.0. We include the original file header below:
"""
Copyright 2023 The Pgx Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple, Tuple
from pydantic import BaseModel

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import wandb
from omegaconf import OmegaConf

# Import the PGX environment
import pgx
from pgx.experimental import auto_reset

# Import the LDX environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import LudaxEnvironment

# CNN module from PGX
from network import AZNet


devices = jax.local_devices()
num_devices = len(devices)

class Config(BaseModel):
    env_id: str = "reversi" # other supported games are "tic_tac_toe", "hex", "connect_four", "reversi", ...
    env_type: str = "ldx"
    seed: int = 0
    max_num_iters: int = 1000
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"

conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)


def pgx_baseline(env_id):
    """
    Load a pgx baseline model for the given environment ID.
    """

    # reversi is called othello for PGX
    if env_id == "reversi":
        env_id = "othello"

    baseline_fn = pgx.make_baseline_model(env_id + "_v0")

    @jax.jit
    def baseline_wrap(state):
        logits, _ = baseline_fn(observe(state))
        return logits

    return baseline_wrap


@jax.jit
def random_baseline(state):
    """
    A random baseline that samples actions uniformly from the legal action mask. Used if no baseline model is available.
    """
    if config.env_type=="pgx":
        return jnp.log(state.legal_action_mask.astype(jnp.float32))

    return jnp.log(state.legal_action_mask.astype(jnp.float16))


@partial(jax.jit, static_argnames=["observation_shape"])
def board_to_observation(board: jnp.ndarray, current_player: jnp.ndarray, observation_shape: Tuple[int, int]) -> jnp.ndarray:
    """
    Convert a flat board with values in {-1, 0, 1} symbolizing the current piece type into a boolean (rows, cols, 2) tensor symbolizing who is playing.
    board[i] == -1  → empty square
    board[i] == 0   → square occupied by white (current player or not)
    board[i] == 1   → square occupied by black (current player or not)
    observation[:, :, 0] == True  → squares occupied by the current player (black or white)
    observation[:, :, 1] == True  → squares occupied by the other player (black or white)
    """
    board2d = board.reshape(*board.shape[:-1], observation_shape[-3], observation_shape[-2])
    current_player = current_player[..., None, None]
    return jnp.stack((board2d == current_player, board2d == jnp.abs(1 - current_player)), axis=-1)


@jax.jit
def observe(state: pgx.State) -> jnp.ndarray:
    """
    Wrapper function to convert the state to the model input regardless of the env_type.
    """
    if config.env_type=="pgx":
        return state.observation
    else:
        return board_to_observation(state.game_state.board, state.game_state.current_player, state.observation.shape)


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)
def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player if config.env_type=="pgx" else state.game_state.current_player
    if not config.env_type == "pgx":
        action = action.astype(jnp.int16)
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, observe(state), is_eval=True)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    rewards = state.rewards
    reward = rewards[jnp.arange(rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state

class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)

        observation = observe(state)

        (logits, value), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        actor = state.current_player if config.env_type=="pgx" else state.game_state.current_player
        keys = jax.random.split(key2, batch_size)
        action = policy_output.action
        if not config.env_type == "pgx":
            action = action.astype(jnp.int16)
        state = jax.vmap(auto_reset(env.step, env.init))(state, action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        rewards = state.rewards
        
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=rewards[jnp.arange(rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss



@partial(jax.pmap, static_broadcasted_argnums=[2])
def evaluate(rng_key, my_model, my_player: int):
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_model_params, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        (my_logits, _), _ = forward.apply(
            my_model_params, my_model_state, observe(state), is_eval=True
        )
        opp_logits = baseline(state)
        current_player = state.current_player if config.env_type=="pgx" else state.game_state.current_player
        is_my_turn = (current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(my_logits.dtype).min)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)

        if not config.env_type == "pgx":
            action = action.astype(jnp.int16)

        state = jax.vmap(env.step)(state, action)
        rewards = state.rewards
        R = R + rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


if __name__ == "__main__":

    ## Setup ##

    # Otherllo is called reversi for LDX
    if config.env_type == "pgx" and config.env_id == "reversi":
        config.env_id = "othello"

    if config.env_type == "ldx" and config.env_id == "othello":
        config.env_id = "reversi"

    # Initialize the environment, either PGX or LDX
    env = pgx.make(config.env_id) if config.env_type == "pgx" else LudaxEnvironment(f"games/{config.env_id}.ldx")

    # Load the baseline model for evaluation
    try:
        baseline = pgx_baseline(config.env_id)
        print(f"Loaded baseline model for {config.env_id}")
    except Exception as e:
        print(f"Failed to load baseline model: {e}")
        baseline = random_baseline


    ### Training ###
    wandb.init(project="pgx-az", config=config.model_dump())

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = observe(dummy_state)
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])
    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if iteration % config.eval_interval == 0:
            # Evaluation
            for player in [0, 1]:
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, num_devices)
                R = evaluate(keys, model, player)
                prefix = f"eval/vs_baseline/player{player+1}"
                log.update({
                    f"{prefix}/avg_R": R.mean().item(),
                    f"{prefix}/win_rate": ((R == 1).sum() / R.size).item(),
                    f"{prefix}/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"{prefix}/lose_rate": ((R == -1).sum() / R.size).item(),
                })

            # Store checkpoints
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )
