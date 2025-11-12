"""Integration tests for TruncationWrapper using Container-based State/Info."""

from functools import cached_property

import jax
import jax.numpy as jnp

from jenv.environment import Environment, InfoContainer, StateContainer
from jenv.spaces import Continuous
from jenv.struct import static_field
from jenv.typing import Key
from jenv.wrappers.truncation_wrapper import TruncationWrapper
from jenv.wrappers.wrapper import Wrapper

# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleEnv(Environment):
    """Simple environment that returns Container-based state/info."""

    def reset(self, key: Key) -> tuple[StateContainer, InfoContainer]:
        env_state = jnp.array(0.0)
        state = StateContainer(env_state=env_state)
        info = InfoContainer(obs=env_state, reward=0.0, terminated=False)
        return state, info

    def step(
        self, state: StateContainer, action: jax.Array
    ) -> tuple[StateContainer, InfoContainer]:
        next_env_state = state.env_state + action
        next_state = StateContainer(env_state=next_env_state)
        info = InfoContainer(obs=next_env_state, reward=float(action), terminated=False)
        return next_state, info

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf, shape=())

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0, shape=())


# ============================================================================
# Tests: TruncationWrapper - Basic Functionality
# ============================================================================


class TestTruncationWrapperBasic:
    """Test TruncationWrapper basic functionality with Container state/info."""

    def test_reset_sets_defaults_and_returns_containers(self):
        env = SimpleEnv()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=10)

        key = jax.random.PRNGKey(0)
        state, info = trunc_wrapper.reset(key)

        assert isinstance(state, StateContainer)
        assert hasattr(state, "steps") and state.steps == 0

        assert isinstance(info, InfoContainer)
        assert hasattr(info, "truncated") and info.truncated is False
        assert jnp.allclose(info.obs, jnp.array(0.0))
        assert info.reward == 0.0

    def test_step_tracks_steps_and_truncation(self):
        env = SimpleEnv()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=3)

        key = jax.random.PRNGKey(0)
        state, _ = trunc_wrapper.reset(key)

        state, info = trunc_wrapper.step(state, jnp.array(0.1))
        assert state.steps == 1 and info.truncated is False

        state, info = trunc_wrapper.step(state, jnp.array(0.2))
        assert state.steps == 2 and info.truncated is False

        state, info = trunc_wrapper.step(state, jnp.array(0.3))
        assert state.steps == 3 and info.truncated is True

    def test_env_state_and_obs_progression(self):
        env = SimpleEnv()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=10)

        key = jax.random.PRNGKey(0)
        state, _ = trunc_wrapper.reset(key)

        state, info = trunc_wrapper.step(state, jnp.array(0.5))
        assert jnp.allclose(state.env_state, jnp.array(0.5))
        assert jnp.allclose(info.obs, jnp.array(0.5))

        state, info = trunc_wrapper.step(state, jnp.array(0.3))
        assert jnp.allclose(state.env_state, jnp.array(0.8))
        assert jnp.allclose(info.obs, jnp.array(0.8))

    def test_reset_resets_step_count(self):
        env = SimpleEnv()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=5)

        key = jax.random.PRNGKey(0)
        state, _ = trunc_wrapper.reset(key)
        state, _ = trunc_wrapper.step(state, jnp.array(0.1))
        state, _ = trunc_wrapper.step(state, jnp.array(0.2))
        assert state.steps == 2

        state, info = trunc_wrapper.reset(key)
        assert state.steps == 0
        assert info.truncated is False

    def test_delegates_spaces_and_unwrapped(self):
        env = SimpleEnv()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=10)

        assert trunc_wrapper.observation_space == env.observation_space
        assert trunc_wrapper.action_space == env.action_space
        assert trunc_wrapper.unwrapped is env


# ============================================================================
# Tests: TruncationWrapper - Nested Wrapping and Variants
# ============================================================================


class TestTruncationWrapperNested:
    def test_nested_with_environment_wrapper(self):
        env = SimpleEnv()
        inner_wrapper = Wrapper(env=env)
        trunc_wrapper = TruncationWrapper(env=inner_wrapper, max_steps=2)

        key = jax.random.PRNGKey(0)
        state, info = trunc_wrapper.reset(key)

        assert isinstance(state, StateContainer)
        assert isinstance(info, InfoContainer)
        assert state.steps == 0 and info.truncated is False
        assert jnp.allclose(info.obs, jnp.array(0.0))

    def test_overrides_existing_truncated_from_base_env(self):
        class EnvWithTruncated(SimpleEnv):
            def step(self, state: StateContainer, action: jax.Array):
                next_state, base_info = super().step(state, action)
                # Base env marks truncated=True, wrapper must overwrite based on step count
                return next_state, base_info.update(truncated=True)

        env = EnvWithTruncated()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=10)

        key = jax.random.PRNGKey(0)
        state, _ = trunc_wrapper.reset(key)
        _, info = trunc_wrapper.step(state, jnp.array(0.5))

        assert hasattr(info, "truncated")
        assert info.truncated is False  # step=1, max_steps=10

    def test_multiple_truncation_wrappers(self):
        env = SimpleEnv()
        inner = TruncationWrapper(env=env, max_steps=3)
        outer = TruncationWrapper(env=inner, max_steps=5)

        key = jax.random.PRNGKey(0)
        state, _ = outer.reset(key)
        for i in range(4):
            state, info = outer.step(state, jnp.array(0.1))
            assert state.steps == i + 1
            if i + 1 >= 5:
                assert info.truncated is True
            else:
                assert info.truncated is False

    def test_with_dict_state(self):
        class DictStateEnv(Environment):
            def reset(self, key: Key):
                env_state = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
                return StateContainer(env_state=env_state), InfoContainer(
                    obs=env_state["x"], reward=0.0, terminated=False
                )

            def step(self, state: StateContainer, action: jax.Array):
                next_env_state = {
                    "x": state.env_state["x"] + action,
                    "y": state.env_state["y"] + action * 2,
                }
                return StateContainer(env_state=next_env_state), InfoContainer(
                    obs=next_env_state["x"], reward=1.0, terminated=False
                )

            @cached_property
            def observation_space(self) -> Continuous:
                return Continuous(low=-jnp.inf, high=jnp.inf, shape=())

            @cached_property
            def action_space(self) -> Continuous:
                return Continuous(low=-1.0, high=1.0, shape=())

        env = DictStateEnv()
        trunc_wrapper = TruncationWrapper(env=env, max_steps=5)

        key = jax.random.PRNGKey(0)
        state, _ = trunc_wrapper.reset(key)
        assert isinstance(state, StateContainer)
        assert isinstance(state.env_state, dict)
        assert "x" in state.env_state and "y" in state.env_state

        state, info = trunc_wrapper.step(state, jnp.array(0.5))
        assert jnp.allclose(state.env_state["x"], jnp.array(0.5))
        assert jnp.allclose(state.env_state["y"], jnp.array(1.0))
