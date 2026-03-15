"""Playwright test configuration."""

import pytest


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for all tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1440, "height": 900},
        "ignore_https_errors": True,
    }
