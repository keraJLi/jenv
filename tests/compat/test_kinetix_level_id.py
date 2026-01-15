"""Tests for Kinetix level id normalization."""

import pytest

from jenv.compat.kinetix_jenv import _normalize_level_id

pytestmark = pytest.mark.compat


def test_normalize_level_id_appends_json():
    assert _normalize_level_id("s/h4_thrust_aim") == "s/h4_thrust_aim.json"


def test_normalize_level_id_keeps_json():
    assert _normalize_level_id("s/h4_thrust_aim.json") == "s/h4_thrust_aim.json"


def test_normalize_level_id_strips_leading_slash_and_whitespace():
    assert _normalize_level_id(" /s/h4_thrust_aim  ") == "s/h4_thrust_aim.json"


@pytest.mark.parametrize("bad", ["", "   ", "/", "s/"])
def test_normalize_level_id_rejects_empty_or_trailing_slash(bad: str):
    with pytest.raises(ValueError):
        _normalize_level_id(bad)
