import jax

from jenv.environment import Info
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper

RESET_KEY_NAME = "reset_key"


class AutoResetWrapper(Wrapper):
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
        state, info = self.env.reset(key)
        episodic = state.episodic.update(**{RESET_KEY_NAME: subkey})
        state = state.update(episodic=episodic)
        return state, info.update(next_obs=info.obs)

    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        state_step, info_step = self.env.step(state, action)
        done = info_step.terminated | info_step.truncated
        reset_key = getattr(state_step.episodic, RESET_KEY_NAME)

        info = info_step.update(next_obs=info_step.obs)
        state_next, info_next = jax.lax.cond(
            done,
            lambda: self.reset(reset_key),
            lambda: (state_step, info),
        )
        return state_next, info_next
