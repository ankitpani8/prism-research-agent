# Sample query

**Question**

> What are the top drivers of prepaid customer complaints this quarter, and what
> interventions should we prioritise?

**Optional visual input:** a prepaid CX dashboard image
(`data/sample_dashboard.png`) — a complaint-category bar chart + an NPS trend
line. Generate it with `python scripts/make_dashboard.py`.

## How to run

CLI (prints the live agent trace, then the brief):

```bash
python run_demo.py "What are the top drivers of prepaid customer complaints this quarter?"
# with the dashboard image (vision agent reads it):
python run_demo.py "What are the top drivers of prepaid customer complaints this quarter?" \
  --image data/sample_dashboard.png
```

Streamlit UI:

```bash
streamlit run streamlit_app.py
```

FastAPI surface (SSE):

```bash
curl -N -X POST localhost:8000/research -H 'content-type: application/json' \
  -d '{"question":"What are the top drivers of prepaid customer complaints this quarter?"}'
```

See [`sample_output.md`](./sample_output.md) for a real, reproducible run.
