import warnings
from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams
from craftax.craftax_classic.envs.craftax_state import (
    EnvParams as CraftaxClassicEnvParams,
)
from craftax.craftax_env import make_craftax_env_from_name

from jenv import spaces as jenv_spaces
from jenv.compat.gymnax_jenv import _convert_space as _convert_gymnax_space
from jenv.environment import Environment, Info, InfoContainer, State
from jenv.struct import Container, static_field
from jenv.typing import Key, PyTree, TypeAlias

EnvParams: TypeAlias = CraftaxEnvParams | CraftaxClassicEnvParams


class CraftaxJenv(Environment):
    """Wrapper to convert a Craftax environment to a jenv environment."""

    craftax_env: Any = static_field()
    env_params: PyTree

    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_params: EnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "CraftaxJenv":
        env_kwargs = env_kwargs or {}
        auto_reset = env_kwargs.setdefault("auto_reset", False)
        if auto_reset:
            warnings.warn(
                "Creating a CraftaxJenv with auto_reset=True is not recommended, use "
                "an AutoResetWrapper instead."
            )

        env = make_craftax_env_from_name(env_name, **env_kwargs)
        default_params = env.default_params.replace(max_timesteps=jnp.inf)

        env_params = env_params or default_params
        if env_params.max_timesteps < jnp.inf:
            warnings.warn(
                "Creating a CraftaxJenv with a finite max_timesteps is not "
                "recommended, use a TruncationWrapper instead."
            )

        return cls(craftax_env=env, env_params=env_params)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.craftax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.craftax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(info=env_info)
        return state, info

    @override
    @cached_property
    def action_space(self) -> jenv_spaces.Space:
        return _convert_gymnax_space(self.craftax_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> jenv_spaces.Space:
        return _convert_gymnax_space(
            self.craftax_env.observation_space(self.env_params)
        )
