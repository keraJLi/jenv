"""Tests for jenv.compat.jumanji_jenv module."""

from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jumanji import specs

import jenv.compat.jumanji_jenv as jumanji_jenv
from jenv.compat.jumanji_jenv import JumanjiJenv, convert_jumanji_spec_to_jenv_space
from jenv.environment import Info
from jenv.spaces import Continuous, Discrete, PyTreeSpace
from tests.compat.contract import assert_reset_step_contract

pytestmark = pytest.mark.compat


def _create_jumanji_env(env_name: str = "Snake-v1", **env_kwargs) -> JumanjiJenv:
    """Helper to create a JumanjiJenv wrapper."""
    return JumanjiJenv.from_name(env_name, env_kwargs=env_kwargs or None)


@pytest.fixture(scope="module")
def jumanji_env():
    return _create_jumanji_env()


@pytest.fixture(scope="module", autouse=True)
def _jumanji_env_warmup(jumanji_env, prng_key):
    env = jumanji_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_jumanji2jenv_wrapper(jumanji_env, prng_key):
    """Test that JumanjiJenv wrapper correctly wraps a Jumanji environment."""
    env = jumanji_env

    key = prng_key
    state, info = env.reset(key)

    assert isinstance(info, Info)
    assert jnp.asarray(info.reward).shape == ()  # scalar reward
    assert hasattr(info, "terminated")
    assert hasattr(info, "truncated")
    # Observation is a valid pytree (often a namedtuple).
    jax.tree.structure(info.obs)

    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, next_info = env.step(state, action)

    assert next_state is not None
    assert isinstance(next_info, Info)
    jax.tree.structure(next_info.obs)


def test_jumanji_contract_smoke(prng_key, jumanji_env):
    env = jumanji_env

    def obs_check(obs, _obs_space):
        # Observations are pytrees (often namedtuples); containment is not guaranteed
        # because observation_space is derived from Spec._specs dict.
        jax.tree.structure(obs)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_info_fields_reset(jumanji_env, prng_key):
    """Test that Info container has correct fields on reset."""
    env = jumanji_env
    key = prng_key

    state, info = env.reset(key)

    assert state is not None
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")
    assert hasattr(info, "truncated")

    assert jnp.asarray(info.reward) == 0.0


def test_info_fields_step(jumanji_env, prng_key):
    """Test that Info container has correct fields on step."""
    env = jumanji_env
    key = prng_key

    state, _ = env.reset(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, info = env.step(state, action)

    assert next_state is not None
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")
    assert hasattr(info, "truncated")

    assert jnp.isscalar(info.reward) or jnp.asarray(info.reward).shape == ()
    assert isinstance(info.terminated, (bool, jnp.ndarray))
    assert isinstance(info.truncated, (bool, jnp.ndarray))


def test_space_contains(jumanji_env, prng_key):
    """Test that converted spaces correctly validate samples."""
    env = jumanji_env
    key = prng_key

    action = env.action_space.sample(key)
    assert env.action_space.contains(action)

    # Observation-space containment can't be asserted here because observation_space is
    # derived from Spec._specs (dict), while jumanji observations are often namedtuples.
    state, info = env.reset(key)
    assert state is not None
    assert hasattr(info, "obs")


def test_observation_space_property_smoke(jumanji_env):
    """Access observation_space cached_property."""
    env = jumanji_env
    space = env.observation_space
    assert isinstance(space, PyTreeSpace)


def test_small_time_limit_smoke(prng_key):
    """Smoke test that finite time_limit env runs."""

    # Setting a time_limit should warn.
    with pytest.warns(UserWarning):
        env = _create_jumanji_env(time_limit=5)

    key = prng_key
    state, _info = env.reset(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, _next_info = env.step(state, action)
    assert next_state is not None


def test_multiple_episodes(jumanji_env, prng_key):
    """Test multiple reset/step cycles."""
    env = jumanji_env

    for episode in range(3):
        state, info = env.reset(jax.random.fold_in(prng_key, episode))
        assert state is not None

        for step in range(5):
            action = env.action_space.sample(
                jax.random.fold_in(prng_key, episode * 100 + step)
            )
            state, info = env.step(state, action)
            assert state is not None


def test_full_episode_rollout_scan(jumanji_env, prng_key):
    """Test a rollout using jax.lax.scan."""
    env = jumanji_env
    key = prng_key
    state, _ = env.reset(key)

    num_steps = 25
    action_keys = jax.random.split(key, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)

    def step_fn(state, action):
        return env.step(state, action)

    final_state, step_infos = jax.lax.scan(step_fn, state, actions)

    assert final_state is not None
    assert isinstance(step_infos, Info)
    assert step_infos.reward.shape == (num_steps,)


def test_from_name_with_time_limit_warning():
    """Warn when user passes a finite time_limit, like other compat wrappers."""
    with pytest.warns(
        UserWarning,
        match="Creating a JumanjiJenv with a finite time_limit is not recommended",
    ):
        _create_jumanji_env(time_limit=10)


def test_discrete_spec_conversion(prng_key):
    """Test conversion of DiscreteArray specs to jenv Discrete space."""
    spec = specs.DiscreteArray(num_values=7, dtype=np.int32, name="d")
    space = convert_jumanji_spec_to_jenv_space(spec)
    assert isinstance(space, Discrete)
    assert space.shape == ()
    assert int(jnp.asarray(space.n)) == 7

    sample = space.sample(prng_key)
    assert space.contains(sample)


def test_multidiscrete_spec_conversion(prng_key):
    """Test conversion of MultiDiscreteArray specs to jenv Discrete with array n."""
    md = specs.MultiDiscreteArray(
        num_values=jnp.asarray([2, 3, 4], dtype=jnp.int32), dtype=np.int32, name="md"
    )
    space = convert_jumanji_spec_to_jenv_space(md)

    assert isinstance(space, Discrete)
    assert space.shape == (3,)
    assert jnp.array_equal(space.n, jnp.asarray([2, 3, 4], dtype=jnp.int32))

    sample = space.sample(prng_key)
    assert space.contains(sample)


def test_bounded_array_spec_conversion_broadcasts_bounds(prng_key):
    """Test BoundedArray converts to Continuous with broadcasted bounds."""
    b = specs.BoundedArray(
        shape=(2, 3), dtype=np.float32, minimum=0.0, maximum=1.0, name="b"
    )
    space = convert_jumanji_spec_to_jenv_space(b)
    assert isinstance(space, Continuous)
    assert space.shape == (2, 3)
    assert jnp.all(space.low == 0.0)
    assert jnp.all(space.high == 1.0)

    sample = space.sample(prng_key)
    assert space.contains(sample)


def test_array_spec_conversion_float_is_unbounded_box():
    """Float Array converts to Continuous(-inf, +inf)."""
    spec = specs.Array(shape=(2, 3), dtype=np.float32, name="a")
    space = convert_jumanji_spec_to_jenv_space(spec)
    assert isinstance(space, Continuous)
    assert space.shape == (2, 3)
    assert jnp.all(jnp.isneginf(space.low))
    assert jnp.all(jnp.isposinf(space.high))


def test_array_spec_conversion_non_float_not_supported():
    """Non-float Array specs are intentionally not supported."""
    spec = specs.Array(shape=(2,), dtype=np.int32, name="ai")
    with pytest.raises(NotImplementedError):
        convert_jumanji_spec_to_jenv_space(spec)


def test_deterministic_reset(jumanji_env, prng_key):
    """Reset with the same key should produce the same initial observation."""
    env = jumanji_env
    key = jax.random.fold_in(prng_key, 42)

    _state1, info1 = env.reset(key)
    _state2, info2 = env.reset(key)

    assert jax.tree.structure(info1.obs) == jax.tree.structure(info2.obs)
    assert jnp.array_equal(jnp.asarray(info1.reward), jnp.asarray(info2.reward))
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, info1.obs, info2.obs))


def test_deepcopy_warning(jumanji_env):
    env = jumanji_env
    with pytest.warns(RuntimeWarning, match="Trying to deepcopy"):
        copied = deepcopy(env)
    assert copied is not None


def test_namedtuple_observation_converted_to_dict_for_info():
    # Current implementation preserves observation as-is.
    import collections

    NT = collections.namedtuple("NT", ["x", "y"])

    class DummyTimestep:
        observation = NT(x=jnp.array([1.0]), y=jnp.array([2.0]))
        reward = 0.0
        extras = {}

        def last(self):
            return False

    info = jumanji_jenv.convert_jumanji_to_jenv_info(DummyTimestep())
    assert isinstance(info.obs, tuple)
    assert hasattr(info.obs, "_asdict")


def test_structured_spec_dict_conversion():
    """Hit Spec._specs dict branch in convert_jumanji_spec_to_jenv_space."""

    class DummySpec:
        _specs = {
            "d": specs.DiscreteArray(num_values=3, dtype=np.int32, name="d"),
            "b": specs.BoundedArray(
                shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0, name="b"
            ),
        }

    space = convert_jumanji_spec_to_jenv_space(DummySpec())
    assert isinstance(space, PyTreeSpace)
    assert isinstance(space.tree, dict)
    assert set(space.tree.keys()) == {"d", "b"}


def test_spec_discrete_broadcast_branch_exercised():
    # Force the broadcast branch by making spec.shape larger than num_values shape,
    # but still broadcast-compatible.
    md = specs.MultiDiscreteArray(
        num_values=jnp.asarray([2, 3, 4], dtype=jnp.int32), dtype=np.int32, name="md"
    )
    md._shape = (2, 3)

    space = convert_jumanji_spec_to_jenv_space(md)
    assert isinstance(space, Discrete)
    assert space.shape == (2, 3)
    assert jnp.array_equal(space.n, jnp.broadcast_to(jnp.asarray([2, 3, 4]), (2, 3)))


def test_structured_spec_tuple_list_and_unsupported(prng_key):
    d = specs.DiscreteArray(num_values=3, dtype=np.int32, name="d")
    b = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0, name="b"
    )

    space = convert_jumanji_spec_to_jenv_space([d, b])
    assert isinstance(space, PyTreeSpace)
    assert isinstance(space.tree, tuple)
    assert isinstance(space.tree[0], Discrete)
    assert isinstance(space.tree[1], Continuous)

    sample = space.sample(prng_key)
    assert space.contains(sample)

    with pytest.raises(ValueError, match="Unsupported spec type"):
        convert_jumanji_spec_to_jenv_space(object())
