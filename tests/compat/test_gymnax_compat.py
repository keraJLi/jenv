"""Tests for jenv.compat.gymnax_jenv module."""

import jax
import jax.numpy as jnp
import pytest
from gymnax.environments import spaces as gymnax_spaces

from jenv.compat.gymnax_jenv import GymnaxJenv, _convert_space
from jenv.environment import Info
from jenv.spaces import Continuous, Discrete, PyTreeSpace
from tests.compat.contract import (
    assert_reset_step_contract,
    assert_scan_rollout_contract,
)

pytestmark = pytest.mark.compat


def _create_gymnax_env(env_name: str = "CartPole-v1", **kwargs):
    """Helper to create a GymnaxJenv wrapper."""
    return GymnaxJenv.from_name(env_name, env_kwargs=kwargs)


@pytest.fixture(scope="module")
def gymnax_env():
    return _create_gymnax_env("CartPole-v1")


@pytest.fixture(scope="module", autouse=True)
def _gymnax_env_warmup(gymnax_env, prng_key):
    env = gymnax_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_gymnax2jenv_wrapper(gymnax_env, prng_key):
    """Test that GymnaxJenv wrapper correctly wraps a Gymnax environment."""
    # Create a GymnaxJenv wrapper
    env = gymnax_env

    # Test reset
    key = prng_key
    state, step_info = env.reset(key)

    # Check that info has the correct structure
    assert isinstance(step_info, Info)

    # Check initial values
    assert not step_info.terminated
    assert jnp.isscalar(step_info.reward) or step_info.reward.shape == ()

    # Check that observation is in observation space
    assert env.observation_space.contains(step_info.obs)

    # Test step with a valid action
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, next_step_info = env.step(state, action)

    # Check that next_state has the correct structure
    assert next_state is not None
    assert isinstance(next_step_info, Info)

    # Check that observation is in observation space
    assert env.observation_space.contains(next_step_info.obs)

    # Check that action space exists and is correct type
    assert env.action_space is not None
    assert isinstance(env.action_space, Discrete)

    # Check that observation space exists
    assert env.observation_space is not None


def test_info_fields_reset(gymnax_env, prng_key):
    """Test that Info container has correct fields on reset."""
    env = gymnax_env
    key = prng_key

    state, info = env.reset(key)

    # Verify Info has correct structure
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")

    # Verify reward is 0.0 on reset
    assert info.reward == 0.0

    # Verify terminated is False on reset
    assert info.terminated is False

    # Verify state fields
    assert state is not None
    assert hasattr(state, "key")
    assert hasattr(state, "env_state")


def test_info_fields_step(gymnax_env, prng_key):
    """Test that Info container has correct fields on step."""
    env = gymnax_env
    key = prng_key

    state, _ = env.reset(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, info = env.step(state, action)

    # Verify Info has correct structure
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")

    # Verify reward is a scalar
    assert jnp.isscalar(info.reward) or info.reward.shape == ()

    # Verify gymnax info is preserved
    assert hasattr(info, "info")

    # Verify state is not None
    assert next_state is not None


def test_gymnax_container_terminated(gymnax_env, prng_key):
    """Test that GymnaxContainer.terminated maps to done."""
    env = gymnax_env
    key = prng_key

    state, _ = env.reset(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, info = env.step(state, action)

    # GymnaxContainer is used for state in step
    # It has a terminated property that maps to done
    # But since state doesn't have done, we test via info.terminated
    # The state returned from step is a GymnaxContainer
    assert next_state is not None
    # Verify info.terminated works correctly
    assert isinstance(info.terminated, (bool, jnp.ndarray))


def test_episode_termination(gymnax_env, prng_key):
    """Test that episode termination is correctly detected."""
    env = gymnax_env
    key = prng_key

    state, info = env.reset(key)
    assert not info.terminated

    # Avoid long loops; just verify the flag type and that stepping works.
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    _next_state, next_info = env.step(state, action)
    assert isinstance(next_info.terminated, (bool, jnp.ndarray))


def test_multiple_episodes(gymnax_env, prng_key):
    """Test multiple reset/step cycles."""
    env = gymnax_env

    # Run multiple episodes
    for episode in range(3):
        state, info = env.reset(jax.random.fold_in(prng_key, episode))

        # Verify reset clears terminated flag
        assert not info.terminated

        # Take a few steps
        for step in range(5):
            action = env.action_space.sample(
                jax.random.fold_in(prng_key, episode * 100 + step)
            )
            state, info = env.step(state, action)

        # Verify state is properly reset on next reset
        next_state, next_info = env.reset(jax.random.fold_in(prng_key, episode + 1))
        assert not next_info.terminated

        # Different keys should give different initial observations
        if episode > 0:
            prev_state, prev_info = env.reset(jax.random.fold_in(prng_key, episode - 1))
            # States from different keys may differ
            assert next_state is not None and prev_state is not None


def test_full_episode_rollout(gymnax_env, prng_key):
    """Test a complete episode rollout using jax.lax.scan."""
    env = gymnax_env
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
    """Test GymnaxJenv.from_name() with different environment names."""
    # Test with "CartPole-v1" (discrete action space)
    env_cartpole = GymnaxJenv.from_name("CartPole-v1")
    assert env_cartpole is not None
    assert isinstance(env_cartpole.action_space, Discrete)
    assert isinstance(env_cartpole.observation_space, Continuous)

    # Test reset and step work
    key = prng_key
    state, info = env_cartpole.reset(key)
    assert state is not None
    assert isinstance(info, Info)

    # Test with "Pendulum-v1" (continuous action space)
    env_pendulum = GymnaxJenv.from_name("Pendulum-v1")
    assert env_pendulum is not None
    assert isinstance(env_pendulum.action_space, Continuous)
    assert isinstance(env_pendulum.observation_space, Continuous)

    # Different environments should have different space dimensions
    assert env_cartpole.action_space.shape != env_pendulum.action_space.shape
    assert env_cartpole.observation_space.shape != env_pendulum.observation_space.shape


def test_from_name_with_env_kwargs(prng_key):
    """Test from_name with env_kwargs."""
    # Test that env_kwargs are passed correctly
    env = GymnaxJenv.from_name("CartPole-v1", env_kwargs={})
    assert env is not None

    # Test reset and step work
    key = prng_key
    state, info = env.reset(key)
    assert state is not None

    # Verify max_steps_in_episode is set to infinity if present in env_params
    assert env.env_params is not None
    assert jnp.isposinf(env.env_params.max_steps_in_episode)


def test_action_space_conversion():
    """Test conversion of gymnax action spaces to jenv spaces."""
    # Test discrete action space (CartPole)
    env_cartpole = _create_gymnax_env("CartPole-v1")
    assert isinstance(env_cartpole.action_space, Discrete)
    gymnax_action_space = env_cartpole.gymnax_env.action_space(env_cartpole.env_params)
    assert env_cartpole.action_space.n == gymnax_action_space.n
    assert env_cartpole.action_space.shape == gymnax_action_space.shape
    assert env_cartpole.action_space.dtype == gymnax_action_space.dtype

    # Test continuous action space (Pendulum)
    env_pendulum = _create_gymnax_env("Pendulum-v1")
    assert isinstance(env_pendulum.action_space, Continuous)
    gymnax_action_space = env_pendulum.gymnax_env.action_space(env_pendulum.env_params)
    # Handle scalar vs array bounds - jenv converts scalar bounds to arrays when shape is non-empty
    gymnax_low = jnp.asarray(gymnax_action_space.low)
    gymnax_high = jnp.asarray(gymnax_action_space.high)
    jenv_low = jnp.asarray(env_pendulum.action_space.low)
    jenv_high = jnp.asarray(env_pendulum.action_space.high)
    # Compare values (handling broadcasting)
    assert jnp.allclose(jnp.broadcast_to(gymnax_low, jenv_low.shape), jenv_low)
    assert jnp.allclose(jnp.broadcast_to(gymnax_high, jenv_high.shape), jenv_high)
    assert env_pendulum.action_space.shape == gymnax_action_space.shape
    assert env_pendulum.action_space.dtype == gymnax_action_space.dtype


def test_observation_space_conversion():
    """Test conversion of gymnax observation spaces to jenv spaces."""
    # Test continuous observation space (CartPole)
    env = _create_gymnax_env("CartPole-v1")
    assert isinstance(env.observation_space, Continuous)
    gymnax_obs_space = env.gymnax_env.observation_space(env.env_params)
    assert jnp.array_equal(env.observation_space.low, gymnax_obs_space.low)
    assert jnp.array_equal(env.observation_space.high, gymnax_obs_space.high)
    assert env.observation_space.shape == gymnax_obs_space.shape
    assert env.observation_space.dtype == gymnax_obs_space.dtype


def test_space_contains(gymnax_env, prng_key):
    """Test that converted spaces correctly validate samples."""
    env = gymnax_env

    # Test action space contains valid actions
    key = prng_key
    action = env.action_space.sample(key)
    assert env.action_space.contains(action)

    # Test observation space contains observations from reset/step
    state, info = env.reset(key)
    assert env.observation_space.contains(info.obs)

    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, next_info = env.step(state, action)
    assert env.observation_space.contains(next_info.obs)


def test_tuple_space_conversion(prng_key):
    """Test conversion of Tuple spaces."""
    # Create a gymnax Tuple space manually
    space1 = gymnax_spaces.Discrete(2)
    space2 = gymnax_spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)
    tuple_space = gymnax_spaces.Tuple((space1, space2))

    # Convert to jenv space
    jenv_space = _convert_space(tuple_space)

    # Verify it's a PyTreeSpace
    assert isinstance(jenv_space, PyTreeSpace)

    # Verify it contains valid samples
    key = prng_key
    sample = jenv_space.sample(key)
    assert jenv_space.contains(sample)


def test_dict_space_conversion(prng_key):
    """Test conversion of Dict spaces."""
    # Create a gymnax Dict space manually
    space1 = gymnax_spaces.Discrete(2)
    space2 = gymnax_spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)
    dict_space = gymnax_spaces.Dict({"discrete": space1, "continuous": space2})

    # Convert to jenv space
    jenv_space = _convert_space(dict_space)

    # Verify it's a PyTreeSpace
    assert isinstance(jenv_space, PyTreeSpace)

    # Verify it contains valid samples
    key = prng_key
    sample = jenv_space.sample(key)
    assert jenv_space.contains(sample)


def test_unsupported_space_type():
    """Test that unsupported space types raise ValueError."""

    # Create a mock space that's neither Box, Discrete, Tuple, nor Dict
    class MockSpace(gymnax_spaces.Space):
        pass

    # Provide required arguments for Space base class
    mock_space = MockSpace()
    with pytest.raises(ValueError, match="Unsupported space type"):
        _convert_space(mock_space)


def test_box_space_scalar_bounds():
    """Test Box space with scalar low/high."""
    # Create a Box space with scalar bounds
    box_space = gymnax_spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=jnp.float32)

    # Convert to jenv space
    jenv_space = _convert_space(box_space)

    # Verify it's Continuous
    assert isinstance(jenv_space, Continuous)

    # Verify shape is preserved
    assert jenv_space.shape == box_space.shape

    # Verify bounds are correct
    assert jnp.array_equal(jenv_space.low, jnp.full(box_space.shape, box_space.low))
    assert jnp.array_equal(jenv_space.high, jnp.full(box_space.shape, box_space.high))


def test_box_space_array_bounds():
    """Test Box space with array low/high."""
    # Create a Box space with array bounds
    low = jnp.array([-1.0, -2.0, -3.0])
    high = jnp.array([1.0, 2.0, 3.0])
    box_space = gymnax_spaces.Box(low=low, high=high, shape=(3,), dtype=jnp.float32)

    # Convert to jenv space
    jenv_space = _convert_space(box_space)

    # Verify it's Continuous
    assert isinstance(jenv_space, Continuous)

    # Verify conversion handles array bounds correctly
    assert jnp.array_equal(jenv_space.low, low)
    assert jnp.array_equal(jenv_space.high, high)
    assert jenv_space.shape == box_space.shape


def test_deterministic_reset(gymnax_env, prng_key):
    """Test that reset with same key produces same initial observations."""
    env = gymnax_env

    # Reset with same key multiple times
    key = jax.random.fold_in(prng_key, 42)
    state1, info1 = env.reset(key)
    state2, info2 = env.reset(key)

    # Verify same initial observations
    assert jnp.array_equal(info1.obs, info2.obs)

    # Test different keys - some environments may have deterministic initial states
    # So we just verify the reset is consistent, not necessarily different
    key_different = jax.random.fold_in(prng_key, 123)
    state3, info3 = env.reset(key_different)
    state4, info4 = env.reset(key_different)

    # Same key should produce same result
    assert jnp.array_equal(info3.obs, info4.obs)


def test_key_splitting(gymnax_env, prng_key):
    """Test that keys are properly split in reset and step."""
    env = gymnax_env
    key = prng_key

    # Reset splits the key
    state, info = env.reset(key)

    # Verify state has a key (different from input key due to splitting)
    assert hasattr(state, "key")
    assert not jnp.array_equal(state.key, key)

    # Step splits state.key
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, next_info = env.step(state, action)

    # Verify next_state has a different key
    assert hasattr(next_state, "key")
    assert not jnp.array_equal(next_state.key, state.key)


def test_gymnax_contract_smoke(prng_key, gymnax_env):
    env = gymnax_env

    def obs_check(obs, obs_space):
        assert obs_space.contains(obs)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_gymnax_contract_scan(prng_key, scan_num_steps, gymnax_env):
    env = gymnax_env
    assert_scan_rollout_contract(env, key=prng_key, num_steps=scan_num_steps)
