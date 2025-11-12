from functools import cached_property
from typing import override

from jenv import spaces
from jenv.environment import Environment, Info, State
from jenv.typing import Key, PyTree


class Wrapper(Environment):
    """Wrapper for environments."""

    env: Environment

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        return self.env.reset(key)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        return self.env.step(state, action)

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return self.env.observation_space

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return self.env.action_space

    @override
    @property
    def unwrapped(self) -> Environment:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)
