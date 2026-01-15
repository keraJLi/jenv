"""Shared contract helpers for compat wrappers.

These functions enforce a consistent baseline across all compat wrappers:
- reset/step return (state, info) with Info fields present
- reward is scalar-ish and finite
- action sampling is valid for action_space
- observation is validated via a provided callback (since some suites have known quirks)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jenv.environment import Info


ObsCheck = Callable[[object, object], None]


def assert_info_has_contract_fields(info: Info) -> None:
    # Protocol adherence: Info is a runtime-checkable Protocol.
    # This is the strongest, most consistent check we can do across wrappers.
    assert isinstance(info, Info)
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")
    assert hasattr(info, "truncated")


def assert_reset_step_contract(
    env,
    *,
    key: jax.Array,
    obs_check: ObsCheck,
) -> None:
    # Derive subkeys so callers can pass a single fixture key.
    key_reset, key_action = jax.random.split(key)

    state, info = env.reset(key_reset)
    assert_info_has_contract_fields(info)
    assert jnp.asarray(info.reward).shape == ()
    assert jnp.isfinite(jnp.asarray(info.reward))
    assert info.reward == 0.0
    assert not bool(jnp.asarray(info.terminated))
    assert not bool(jnp.asarray(info.truncated))

    obs_check(info.obs, env.observation_space)

    action = env.action_space.sample(key_action)
    assert env.action_space.contains(action)
    next_state, next_info = env.step(state, action)
    assert next_state is not None
    assert_info_has_contract_fields(next_info)
    assert jnp.asarray(next_info.reward).shape == ()
    assert jnp.isfinite(jnp.asarray(next_info.reward))
    obs_check(next_info.obs, env.observation_space)


def assert_scan_rollout_contract(
    env,
    *,
    key: jax.Array,
    num_steps: int,
) -> None:
    state, _ = env.reset(key)
    action_keys = jax.random.split(key, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)

    def step_fn(s, a):
        return env.step(s, a)

    final_state, infos = jax.lax.scan(step_fn, state, actions)
    assert final_state is not None
    assert_info_has_contract_fields(infos)
    assert infos.reward.shape == (num_steps,)
    assert jnp.all(jnp.isfinite(jnp.asarray(infos.reward)))

