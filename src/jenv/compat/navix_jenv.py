import dataclasses
from functools import cached_property
from typing import override

import jax.numpy as jnp
import navix
from navix import spaces as navix_spaces
from navix.environments.environment import Environment as NavixEnv

from jenv import spaces as jenv_spaces
from jenv.environment import Environment, Info, InfoContainer, State
from jenv.typing import Key, PyTree


class NavixJenv(Environment):
    navix_env: NavixEnv

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        timestep = self.navix_env.reset(key)
        return timestep, _convert_container(timestep)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        timestep = self.navix_env.step(state, action)
        return timestep, _convert_container(timestep)

    @override
    @cached_property
    def action_space(self) -> jenv_spaces.Space:
        return _convert_space(self.navix_env.action_space)

    @override
    @cached_property
    def observation_space(self) -> jenv_spaces.Space:
        return _convert_space(self.navix_env.observation_space)


def _convert_container(nvx_container: navix.Timestep) -> InfoContainer:
    timestep_dict = dataclasses.asdict(nvx_container)
    step_type = timestep_dict.pop("step_type")
    info = InfoContainer(
        obs=timestep_dict.pop("observation"),
        reward=timestep_dict.pop("reward"),
        terminated=step_type == navix.StepType.TERMINATION,
        truncated=step_type == navix.StepType.TRUNCATION,
    )
    info = info.update(**timestep_dict)
    return info


def _convert_space(nvx_space: navix_spaces.Space) -> jenv_spaces.Space:
    if isinstance(nvx_space, navix_spaces.Discrete):
        n = jnp.asarray(nvx_space.n).astype(nvx_space.dtype)
        return jenv_spaces.Discrete.from_shape(n, shape=nvx_space.shape)

    elif isinstance(nvx_space, navix_spaces.Continuous):
        low = jnp.asarray(nvx_space.minimum).astype(nvx_space.dtype)
        high = jnp.asarray(nvx_space.maximum).astype(nvx_space.dtype)
        return jenv_spaces.Continuous.from_shape(low, high, shape=nvx_space.shape)

    raise ValueError(f"Unsupported space type: {type(nvx_space)}")
