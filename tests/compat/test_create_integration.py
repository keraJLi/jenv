"""Integration tests for jenv.compat.create().

These require optional compatibility dependencies (brax/gymnax/navix). They are
kept separate from the unit tests so a minimal install can still run the suite.
"""

import pytest

from jenv.compat import create
from jenv.environment import Environment

pytestmark = pytest.mark.integration


def test_create_brax_smoke():
    pytest.importorskip("brax")
    import jax

    from jenv.compat.brax_jenv import BraxJenv

    env = create("brax::fast")
    assert isinstance(env, BraxJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(jax.random.PRNGKey(0))
    assert hasattr(info, "obs")


def test_create_gymnax_smoke():
    pytest.importorskip("gymnax")
    import jax

    from jenv.compat.gymnax_jenv import GymnaxJenv

    env = create("gymnax::CartPole-v1")
    assert isinstance(env, GymnaxJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(jax.random.PRNGKey(0))
    assert hasattr(info, "obs")


def test_create_navix_smoke():
    pytest.importorskip("navix")
    import jax

    from jenv.compat.navix_jenv import NavixJenv

    env = create("navix::Navix-Empty-5x5-v0")
    assert isinstance(env, NavixJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(jax.random.PRNGKey(0))
    assert hasattr(info, "obs")
