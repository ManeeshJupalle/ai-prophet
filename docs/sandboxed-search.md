# Search Tools

```python
import os

from ai_prophet.search import SearchClient

search = SearchClient(provider="exa", api_key=os.environ["EXA_API_KEY"])
try:
    results = search.search("NBA playoff picture", limit=3)
finally:
    search.close()
```

Providers: `brave`, `exa`, `tavily`, `perplexity`.

Use `as_of` to apply a cutoff:

```python
results = search.search("NBA playoff picture", limit=3, as_of="2026-05-01")
```

Returned items are dictionaries:

```python
{
    "url": "...",
    "title": "...",
    "snippet": "...",
    "text": "...",
    "score": 1.0,
    "provider": "exa",
    "published_date": "2024-12-31",
    "updated_date": None,
    "crawled_date": None,
    "sandbox_status": "accepted",
    "sandbox_reason": None,
}
```

Cutoff rejections from the last call are exposed on the client:

```python
search.last_rejected
search.last_warnings
```

`ai_prophet.trade.search.SearchClient` remains available as a compatibility
import. New code should import from `ai_prophet.search`.
