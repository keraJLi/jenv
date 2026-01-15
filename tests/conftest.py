import pytest


_COMPAT_IMPORTS: tuple[str, ...] = (
    "brax",
    "gymnax",
    "craftax",
    "navix",
    "jumanji",
    "kinetix",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-compat",
        action="store_true",
        default=False,
        help="Run tests that require optional compat dependencies (brax/gymnax/navix/jumanji/kinetix/craftax).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "compat: tests requiring optional compat dependencies (opt-in via --run-compat)",
    )
    config.addinivalue_line(
        "markers",
        "integration: integration tests (subset of compat in this repository)",
    )

    if config.getoption("--run-compat"):
        import importlib

        missing: list[str] = []
        for mod in _COMPAT_IMPORTS:
            try:
                importlib.import_module(mod)
            except Exception:  # ImportError + any transitive import failures
                missing.append(mod)

        if missing:
            raise pytest.UsageError(
                "Requested compat tests via --run-compat, but some optional "
                f"compat dependencies are missing/broken: {missing}. "
                "Install the full compat dependency group (e.g. `uv sync --group compat`)."
            )


def _is_compat_path(item: pytest.Item) -> bool:
    # Gate all tests under tests/compat/ by default, even if some are dependency-free.
    # This keeps the suite behavior simple and predictable.
    try:
        return "tests/compat/" in str(item.fspath)
    except Exception:  # pragma: no cover
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-compat"):
        return

    skip = pytest.mark.skip(reason="compat tests are opt-in; pass --run-compat to run them")
    for item in items:
        if _is_compat_path(item) or item.get_closest_marker("compat") is not None:
            item.add_marker(skip)

