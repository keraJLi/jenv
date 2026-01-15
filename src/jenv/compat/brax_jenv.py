import dataclasses
import warnings
from copy import copy
from functools import cached_property
from typing import Any, override

from brax.envs import Env as BraxEnv
from brax.envs import Wrapper as BraxWrapper
from brax.envs import create as brax_create
from jax import numpy as jnp

from jenv import spaces
from jenv.environment import Environment, Info, InfoContainer, State
from jenv.struct import static_field
from jenv.typing import Key, PyTree


class BraxJenv(Environment):
    """Wrapper to convert a Brax environment to a jenv environment."""

    brax_env: BraxEnv = static_field()

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "BraxJenv":
        env_kwargs = env_kwargs or {}
        auto_reset = env_kwargs.setdefault("auto_reset", False)
        if auto_reset:
            warnings.warn(
                "Creating a BraxJenv with auto_reset=True is not recommended, use "
                "an AutoResetWrapper instead."
            )

        episode_length = env_kwargs.setdefault("episode_length", jnp.inf)
        if episode_length < jnp.inf:
            warnings.warn(
                "Creating a BraxJenv with a finite episode_length is not recommended, "
                "as brax does not differentiate between truncation and termination. "
            )

        env = brax_create(env_name, **env_kwargs)
        return cls(brax_env=env)

    def __post_init__(self) -> "BraxJenv":
        if isinstance(self.brax_env, BraxWrapper):
            warnings.warn(
                "Environment wrapping should be handled by jenv. "
                "Unwrapping brax environment before converting..."
            )
            object.__setattr__(self, "brax_env", self.brax_env.unwrapped)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        brax_state = self.brax_env.reset(key)
        info = InfoContainer(obs=brax_state.obs, reward=0.0, terminated=False)
        info = info.update(**dataclasses.asdict(brax_state))
        return brax_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        brax_state = self.brax_env.step(state, action)
        info = InfoContainer(
            obs=brax_state.obs, reward=brax_state.reward, terminated=brax_state.done
        )
        info = info.update(**dataclasses.asdict(brax_state))
        return brax_state, info

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        # All brax environments have action limit of -1 to 1
        return spaces.Continuous.from_shape(
            low=-1.0, high=1.0, shape=(self.brax_env.action_size,)
        )

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        # All brax environments have observation limit of -inf to inf
        return spaces.Continuous.from_shape(
            low=-jnp.inf, high=jnp.inf, shape=(self.brax_env.observation_size,)
        )

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a brax env. "
            "Brax envs throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)
