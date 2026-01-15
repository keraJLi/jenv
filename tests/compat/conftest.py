import pytest


@pytest.fixture(scope="module")
def prng_key():
    import jax

    return jax.random.PRNGKey(0)


@pytest.fixture
def scan_num_steps() -> int:
    """Default number of steps for scan-based rollout tests."""
    return 25


@pytest.fixture
def rollout_scan():
    import jax

    def _rollout(env, key, num_steps: int = 5):
        key, reset_key = jax.random.split(key)
        state, _ = env.reset(reset_key)

        action_keys = jax.random.split(key, num_steps)
        actions = jax.vmap(env.action_space.sample)(action_keys)

        def step_fn(state, action):
            return env.step(state, action)

        final_state, infos = jax.lax.scan(step_fn, state, actions)
        return final_state, infos

    return _rollout
