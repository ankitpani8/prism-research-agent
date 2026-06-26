# Sample output

A **real, reproducible** run via the FastAPI `/research` SSE endpoint. This run
used **only the local `qwen2.5:1.5b` model** (every hosted provider was
unavailable — Gemini quota exhausted, no Anthropic credit), which is exactly the
graceful-degradation story: the full multi-agent pipeline completes on a 986 MB
local model at **$0.00**. With Gemini reachable, the `vision`, `light`, and
`heavy` roles bind to hosted models and the brief is richer (dashboard-derived
category percentages and NPS appear as grounded evidence).

## Agent trace (streamed)

```
event: trace  planner    running   decomposing question
event: trace  planner    complete  2 subtasks [web, rag]
event: trace  researcher running   2 subtasks fanning out (async)
event: trace  researcher complete  2 findings (2 with sources)
event: trace  critic     running   validating 2 claims (local model + lexical check)
event: trace  critic     complete  validated 2 claims, 0 flagged (conf 0.95)
event: trace  summariser running   composing structured brief
event: trace  summariser complete  2 evidence items
event: final  <brief below>
```

## Final brief

```json
{
  "question": "What are the top drivers of prepaid customer complaints this quarter?",
  "themes": [
    "Benchmarking Tool for Customer Complaint Analysis",
    "Billing Issues as a Major Driver"
  ],
  "evidence": [
    {
      "claim": "A benchmarking tool exists to analyze customer complaint data related to financial products and services.",
      "source": "https://www.ifsqn.com/forum/index.php/topic/24226-customer-complaints-benchmarking",
      "confidence": 0.9
    },
    {
      "claim": "Billing issues are consistently the single largest driver of complaints in both public regulator disclosures and company documents.",
      "source": "https://blog.lnsresearch.com/quality-metrics-customer-complaints-benchmark-data",
      "confidence": 1.0
    }
  ],
  "confidence": 0.95,
  "flagged_uncertainties": [],
  "recommended_actions": []
}
```

## Telemetry (per-request, real)

```json
{
  "request_id": "bff92b47",
  "llm_calls": 6,
  "cost_usd": 0.0,
  "duration_ms": 87288.3,
  "by_model": { "qwen2.5:1.5b": { "calls": 6, "in": 2168, "out": 529, "usd": 0.0 } },
  "replan_count": 0
}
```

**Reading the result:** two claims, both grounded (the critic flagged none),
overall confidence 0.95, **0 re-plans**, **$0.00** — the local-critic cost lever,
literal. The second claim grounds "billing is the single largest driver" against
the internal corpus; the first is a weaker web-sourced benchmark (0.90). On CPU
the run takes ~87s; hosted models bring this down by an order of magnitude.
