from typing import override

from jenv.environment import Info
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import FrozenWrappedState, WrappedState, Wrapper


class CanonicalizeWrapper(Wrapper):
    """
    Ensures the externally visible state is a WrappedState by wrapping the inner
    environment's native state into `core` on reset, and unwrapping it on step.
    This wrapper should be applied first in a wrapper stack.
    """

    @override
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        state, info = self.env.reset(key)
        state = FrozenWrappedState(core=state)
        return state, info

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        state_core, info = self.env.step(state.core, action)
        state = state.update(core=state_core)
        return state, info
