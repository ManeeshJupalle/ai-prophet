from __future__ import annotations

from unittest.mock import MagicMock

from ai_prophet.search.client import SearchClient
from ai_prophet.trade.agent.pipeline import AgentPipeline


def test_agent_pipeline_close_closes_search_client():
    llm_client = MagicMock()
    api_client = MagicMock()
    search_client = MagicMock()

    pipeline = AgentPipeline(
        llm_client=llm_client,
        event_store=None,
        api_client=api_client,
        config={"search_client": search_client},
    )

    pipeline.close()

    api_client.close.assert_called_once()
    llm_client.close.assert_called_once()
    search_client.close.assert_called_once()


def test_search_client_close_is_idempotent():
    client = SearchClient(api_key="test-key")
    assert client.provider_name == "brave"
    client.close()
    client.close()

    assert client._loop is None
