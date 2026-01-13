import dataclasses
from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
from gymnax import make as gymnax_create
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnv

from jenv import spaces as jenv_spaces
from jenv.environment import Environment, Info, InfoContainer, State
from jenv.struct import Container, static_field
from jenv.typing import Key, PyTree


class GymnaxJenv(Environment):
    """Wrapper to convert a Gymnax environment to a jenv environment."""

    gymnax_env: GymnaxEnv = static_field()
    env_params: PyTree

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None, **kwargs
    ) -> "GymnaxJenv":
        env_kwargs = env_kwargs or {}
        gymnax_env, env_params = gymnax_create(env_name, **env_kwargs)

        # Overwrite episode_length to infinity to avoid truncation. Normally we would
        # issue a warning, but gymnax does not allow setting the maximum episode length
        # Via any arguments to it's make function. So we are never overwriting a user-
        # specified value.
        for field in dataclasses.fields(env_params):
            if field.name == "max_steps_in_episode":
                env_params = env_params.replace(max_steps_in_episode=jnp.inf)

        return cls(gymnax_env=gymnax_env, env_params=env_params, **kwargs)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.gymnax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(info=None)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.gymnax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(info=env_info)
        return state, info

    @override
    @cached_property
    def action_space(self) -> jenv_spaces.Space:
        return _convert_space(self.gymnax_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> jenv_spaces.Space:
        return _convert_space(self.gymnax_env.observation_space(self.env_params))


def _convert_space(gmx_space: gymnax_spaces.Space) -> jenv_spaces.Space:
    if isinstance(gmx_space, gymnax_spaces.Box):
        low = jnp.broadcast_to(gmx_space.low, gmx_space.shape).astype(gmx_space.dtype)
        high = jnp.broadcast_to(gmx_space.high, gmx_space.shape).astype(gmx_space.dtype)
        return jenv_spaces.Continuous(low=low, high=high)
    elif isinstance(gmx_space, gymnax_spaces.Discrete):
        n = jnp.broadcast_to(gmx_space.n, gmx_space.shape).astype(gmx_space.dtype)
        return jenv_spaces.Discrete(n=n)
    elif isinstance(gmx_space, gymnax_spaces.Tuple):
        spaces = tuple(_convert_space(space) for space in gmx_space.spaces)
        return jenv_spaces.PyTreeSpace(spaces)
    elif isinstance(gmx_space, gymnax_spaces.Dict):
        spaces = {k: _convert_space(space) for k, space in gmx_space.spaces.items()}
        return jenv_spaces.PyTreeSpace(spaces)
    raise ValueError(f"Unsupported space type: {type(gmx_space)}")
