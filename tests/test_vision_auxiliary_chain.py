"""Tests for get_vision_auxiliary_client resolution chain.

Regression: the vision auxiliary client was missing _resolve_api_key_provider
from its auto-detection chain. Users with only an Anthropic (or other direct
API-key provider) key got (None, None) from get_vision_auxiliary_client while
get_auxiliary_client worked fine for text tasks.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all provider keys so each test controls its own env."""
    for var in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "NOUS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_vision_falls_back_to_api_key_provider(monkeypatch):
    """When only an API-key provider (e.g. Anthropic) is configured,
    get_vision_auxiliary_client should still return a client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    with patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        from agent.auxiliary_client import get_vision_auxiliary_client

        client, model = get_vision_auxiliary_client()

    assert client is not None, (
        "Vision client should resolve via _resolve_api_key_provider "
        "when only an API-key provider is available"
    )
    assert model is not None


def test_vision_prefers_openrouter_over_api_key(monkeypatch):
    """OpenRouter should be tried before falling back to API-key providers."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    with patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        from agent.auxiliary_client import get_vision_auxiliary_client

        client, model = get_vision_auxiliary_client()

    assert client is not None
    # Should have picked OpenRouter (first in chain), not Anthropic
    call_kwargs = mock_openai.call_args
    assert "openrouter" in str(call_kwargs).lower()


def test_vision_returns_none_when_no_providers(monkeypatch):
    """With no API keys at all, vision should return (None, None) gracefully."""
    from agent.auxiliary_client import get_vision_auxiliary_client

    client, model = get_vision_auxiliary_client()

    assert client is None
    assert model is None
