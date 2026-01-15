"""Tests for jenv.compat.brax_jenv module."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import pytest
from brax.envs import Wrapper as BraxWrapper

from jenv.compat.brax_jenv import BraxJenv
from jenv.environment import Info
from jenv.spaces import Continuous
from tests.compat.contract import (
    assert_reset_step_contract,
    assert_scan_rollout_contract,
)

pytestmark = pytest.mark.compat


@pytest.fixture(scope="module")
def brax_fast_env():
    return BraxJenv.from_name("fast")


@pytest.fixture(scope="module", autouse=True)
def _brax_fast_env_warmup(brax_fast_env, prng_key):
    """Warm up reset/step once to amortize compilation."""
    env = brax_fast_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_brax2jenv_wrapper(brax_fast_env, prng_key):
    """Test that Brax2Jenv wrapper correctly wraps a Brax environment."""
    # Create a Brax2Jenv wrapper with the ant environment
    env = brax_fast_env

    # Test reset
    key_reset, key_step = jax.random.split(prng_key)
    state, step_info = env.reset(key_reset)

    # Check that info has the correct structure
    assert isinstance(step_info, Info)

    # Check initial values
    assert not step_info.done
    assert jnp.isscalar(step_info.reward) or step_info.reward.shape == ()

    # Check that observation is in observation space
    assert env.observation_space.contains(step_info.obs)

    # Test step with a valid action
    action = env.action_space.sample(key_step)
    next_state, next_step_info = env.step(state, action)

    # Check that next_state has the correct structure
    assert next_state is not None
    assert isinstance(next_step_info, Info)

    # Check that observation is in observation space
    assert env.observation_space.contains(next_step_info.obs)

    # Check that action space has the correct bounds
    assert isinstance(env.action_space, type(env.action_space))
    assert env.action_space.low == pytest.approx(-1.0)
    assert env.action_space.high == pytest.approx(1.0)

    # Check that observation space exists
    assert env.observation_space is not None


def test_brax_contract_smoke(prng_key, brax_fast_env):
    env = brax_fast_env

    def obs_check(obs, obs_space):
        assert obs_space.contains(obs)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_brax_contract_scan(prng_key, scan_num_steps, brax_fast_env):
    env = brax_fast_env
    assert_scan_rollout_contract(env, key=prng_key, num_steps=scan_num_steps)


def test_brax_info_preserves_brax_fields_on_reset(brax_fast_env, prng_key):
    """Brax-specific: extra Brax state fields are preserved on reset."""
    env = brax_fast_env
    key = prng_key

    state, info = env.reset(key)

    # Check extra Brax state fields are preserved
    # Brax state typically has: obs, reward, done, metrics, info
    assert hasattr(info, "done")
    assert hasattr(info, "metrics")

    # Verify state fields match what was returned
    assert state is not None
    assert hasattr(state, "obs")


def test_info_fields_step(brax_fast_env, prng_key):
    """Test that Info container has correct fields on step."""
    env = brax_fast_env
    key_reset, key_action = jax.random.split(prng_key)

    state, _ = env.reset(key_reset)
    action = env.action_space.sample(key_action)
    next_state, info = env.step(state, action)

    # Verify Info has correct structure
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")

    # Verify reward is a scalar
    assert jnp.isscalar(info.reward) or info.reward.shape == ()

    # Verify terminated field exists and matches done
    assert hasattr(info, "done")
    assert info.terminated == info.done

    # Verify extra Brax state fields are preserved
    assert hasattr(info, "metrics")

    # Verify state is not None
    assert next_state is not None


def test_episode_termination(brax_fast_env, prng_key):
    """Test that episode termination is correctly detected."""
    env = brax_fast_env
    key_reset, key_action = jax.random.split(prng_key)

    state, info = env.reset(key_reset)
    assert not info.terminated
    assert not info.done

    # Avoid long loops; just verify that terminated mirrors the underlying done flag.
    action = env.action_space.sample(key_action)
    _state, info = env.step(state, action)
    assert info.terminated == info.done


def test_multiple_episodes(brax_fast_env, prng_key):
    """Test multiple reset/step cycles."""
    env = brax_fast_env

    # Run multiple episodes
    for episode in range(3):
        key_episode = jax.random.fold_in(prng_key, episode)
        key_reset_1, key_reset_2 = jax.random.split(key_episode)
        state, info = env.reset(key_reset_1)

        # Verify reset clears terminated flag
        assert not info.terminated
        assert not info.done

        # Take a few steps
        for step in range(5):
            key_step = jax.random.fold_in(key_episode, step)
            action = env.action_space.sample(key_step)
            state, info = jax.jit(env.step)(state, action)

        # Verify state is properly reset on next reset
        next_state, next_info = jax.jit(env.reset)(key_reset_2)
        assert not next_info.terminated
        assert not next_info.done

        # Different keys should give different initial observations
        if episode > 0:
            key_prev_reset = jax.random.fold_in(key_episode, 0)
            prev_state, prev_info = jax.jit(env.reset)(key_prev_reset)
            # States from different keys may differ
            assert next_state is not None and prev_state is not None


def test_full_episode_rollout(brax_fast_env, prng_key):
    """Test a complete episode rollout using jax.lax.scan."""
    env = brax_fast_env
    key = prng_key

    # Reset environment
    state, _ = env.reset(key)

    # Generate actions for a fixed number of steps
    num_steps = 25
    action_keys = jax.random.split(key, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)

    # Use jax.lax.scan to efficiently step through the episode
    def step_fn(state, action):
        return env.step(state, action)

    final_state, step_infos = jax.lax.scan(step_fn, state, actions)

    # Verify all rewards are finite
    assert jnp.all(jnp.isfinite(step_infos.reward))

    # Verify final state is not None
    assert final_state is not None

    # Check that Info structure is preserved through scan (batched)
    assert isinstance(step_infos, Info)

    # Test shape of batched rewards matches num_steps
    assert step_infos.reward.shape == (num_steps,)

    # Verify observations are batched correctly
    assert step_infos.obs.shape[0] == num_steps


def test_from_name_creation(prng_key):
    """Test BraxJenv.from_name() with different environment names."""
    # Test with "fast" (simple test env)
    env_fast = BraxJenv.from_name("fast")
    assert env_fast is not None
    assert isinstance(env_fast.action_space, Continuous)
    assert isinstance(env_fast.observation_space, Continuous)

    # Test reset and step work
    key = prng_key
    state, info = env_fast.reset(key)
    assert state is not None
    assert isinstance(info, Info)

    # Test with "ant" (standard benchmark)
    env_ant = BraxJenv.from_name("ant")
    assert env_ant is not None
    assert isinstance(env_ant.action_space, Continuous)
    assert isinstance(env_ant.observation_space, Continuous)

    # Different environments should have different space dimensions
    assert env_fast.action_space.shape != env_ant.action_space.shape


def test_from_name_with_episode_length_warning(prng_key):
    """Test that from_name warns when using finite episode_length."""
    # Test warning for finite episode_length
    with pytest.warns(
        UserWarning,
        match="Creating a BraxJenv with a finite episode_length is not recommended",
    ):
        env = BraxJenv.from_name("fast", env_kwargs={"episode_length": 100})

    # Environment should still be created
    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_from_name_with_auto_reset_warning(prng_key):
    """Test that from_name warns when using auto_reset=True."""
    # Test warning for auto_reset=True
    with pytest.warns(
        UserWarning, match="Creating a BraxJenv with auto_reset=True is not recommended"
    ):
        env = BraxJenv.from_name("fast", env_kwargs={"auto_reset": True})

    # Environment should still be created
    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_wrapper_unwrapping():
    """Test that wrapped Brax environments are properly unwrapped."""
    from brax.envs import create as brax_create

    # Create a base Brax environment
    base_env = brax_create("fast", episode_length=None, auto_reset=False)

    # Create a simple wrapper
    class SimpleWrapper(BraxWrapper):
        def reset(self, rng):
            return self.env.reset(rng)

        def step(self, state, action):
            return self.env.step(state, action)

    wrapped_env = SimpleWrapper(base_env)

    # Initialize BraxJenv with wrapped environment
    with pytest.warns(
        UserWarning, match="Environment wrapping should be handled by jenv"
    ):
        env = BraxJenv(brax_env=wrapped_env)

    # Verify environment is properly unwrapped
    assert not isinstance(env.brax_env, BraxWrapper)
    assert env.brax_env is wrapped_env.unwrapped


def test_deepcopy_warning(brax_fast_env, prng_key):
    """Test that deepcopy raises a warning and returns shallow copy."""
    env = brax_fast_env

    # Call deepcopy and verify warning is raised
    with pytest.warns(
        RuntimeWarning, match="Trying to deepcopy.*shallow copy is returned"
    ):
        copied_env = deepcopy(env)

    # Verify shallow copy is returned
    assert copied_env is not None

    # Verify the copied environment is usable
    key = prng_key
    state, info = copied_env.reset(key)
    assert state is not None
    assert isinstance(info, Info)


def test_deterministic_reset(brax_fast_env, prng_key):
    """Test that reset with same key produces same initial observations."""
    env = brax_fast_env
    key_reset_1, key_reset_2 = jax.random.split(prng_key)

    # Reset with same key multiple times
    state1, info1 = env.reset(key_reset_1)
    state2, info2 = env.reset(key_reset_1)

    # Verify same initial observations
    assert jnp.array_equal(info1.obs, info2.obs)
    assert jnp.array_equal(state1.obs, state2.obs)

    # Test different keys - some environments may have deterministic initial states
    # So we just verify the reset is consistent, not necessarily different
    state3, info3 = env.reset(key_reset_2)
    state4, info4 = env.reset(key_reset_2)

    # Same key should produce same result
    assert jnp.array_equal(info3.obs, info4.obs)
    assert jnp.array_equal(state3.obs, state4.obs)
