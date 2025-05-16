# NOTE: This file is largely copied from PGX (https://github.com/sotetsuk/pgx/blob/main/pgx/core.py). Some
# code has been removed that is not relevant to the Ludax framework
# Copyright belongs to the original authors
# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Literal, Optional, Tuple, get_args

import jax
import jax.numpy as jnp

from config import Array, PRNGKey, TRUE, FALSE
from pgx_struct import dataclass

@dataclass
class State(abc.ABC):
    """Base state class of all Pgx game environments. Basically an immutable (frozen) dataclass.
    A basic usage is generating via `Env.init`:

        state = env.init(jax.random.PRNGKey(0))

    and `Env.step` receives and returns this state class:

        state = env.step(state, action)

    Serialization via `flax.struct.serialization` is supported.
    There are 6 common attributes over all games:

    Attributes:
        current_player (Array): id of agent to play.
            Note that this does NOT represent the turn (e.g., black/white in Go).
            This ID is consistent over the parallel vmapped states.
        observation (Array): observation for the current state.
            `Env.observe` is called to compute.
        rewards (Array): the `i`-th element indicates the intermediate reward for
            the agent with player-id `i`. If `Env.step` is called for a terminal state,
            the following `state.rewards` is zero for all players.
        terminated (Array): denotes that the state is terminal state. Note that
            some environments (e.g., Go) have an `max_termination_steps` parameter inside
            and will terminate within a limited number of states (following AlphaGo).
        truncated (Array): indicates that the episode ends with the reason other than termination.
            Note that current Pgx environments do not invoke truncation but users can use `TimeLimit` wrapper
            to truncate the environment. In Pgx environments, some MinAtar games may not terminate within a finite timestep.
            However, the other environments are supposed to terminate within a finite timestep with probability one.
        legal_action_mask (Array): Boolean array of legal actions. If illegal action is taken,
            the game will terminate immediately with the penalty to the palyer.
    """

    current_player: Array
    observation: Array
    rewards: Array
    terminated: Array
    truncated: Array
    legal_action_mask: Array
    _step_count: Array

class Env(abc.ABC):
    def __init__(self): ...

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.

        Returns:
            State: initial state of environment

        """
        state = self._init(key)
        observation = self.observe(state, state.game_state.current_player)
        return state.replace(observation=observation)  # type: ignore

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.game_state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),  # type: ignore
            lambda: self._step(state, action, key),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )

        observation = self.observe(state, state.game_state.current_player)
        state = state.replace(observation=observation)  # type: ignore

        return state

    def observe(self, state: State, player_id: Array) -> Array:
        """Observation function."""
        obs = self._observe(state, player_id)
        return jax.lax.stop_gradient(obs)

    @abc.abstractmethod
    def _init(self, key: PRNGKey) -> State:
        """Implement game-specific init function here."""
        ...

    @abc.abstractmethod
    def _step(self, state, action, key) -> State:
        """Implement game-specific step function here."""
        ...

    @abc.abstractmethod
    def _observe(self, state: State, player_id: Array) -> Array:
        """Implement game-specific observe function here."""
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        ...

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = self.init(jax.random.PRNGKey(0))
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self._observe(state, state.current_player)
        return obs.shape

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return state.replace(rewards=reward, terminated=TRUE)  # type: ignore