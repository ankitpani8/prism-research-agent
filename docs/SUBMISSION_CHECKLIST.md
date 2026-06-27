# Submission checklist

Maps to the assessment deliverables and the final pre-submit steps.

## Deliverables (assignment → where)
- [x] **Source code + README** with setup/deps/env — `README.md`, `requirements.txt`
- [x] **Architecture overview** (diagram + written, roles/interactions) — `docs/architecture.md`, `docs/diagram.mmd`
- [x] **Sample input & output** — `examples/sample_query.md`, `examples/sample_output.md`
- [ ] **Screen recording (3–5 min, MANDATORY)** — follow `docs/recording_runsheet.md`
- [x] Multimodal used in research *and* output — vision agent → grounded brief evidence
- [x] Guardrails/eval catching something — critic flags + `eval/run_eval.py` regression
- [x] CX framing written up — `docs/architecture.md` §8
- [x] Decoupled `core/` callable from CLI / Streamlit / FastAPI
- [x] Containerised — `Dockerfile` + `docker-compose.yml`
- [x] Tests + CI green without live keys — `tests/`, `.github/workflows/ci.yml`
- [ ] Public repo clean — no `.env` / `CLAUDE.md` / keys in history (run the scan below)

## Final pre-submit steps
```bash
# 1. everything committed?
git status                      # should be clean

# 2. the rail is green, keyless
ruff check . && pytest -q       # 38 passed
python eval/run_eval.py         # [eval] PASS

# 3. no secrets tracked or in history
bash scripts/scan_secrets.sh    # >> secret scan CLEAN.

# 4. tag a release
git tag -a v1.0 -m "Prism v1.0 — multi-agent research assistant"
git push origin v1.0

# 5. package the source (only committed files; secrets auto-excluded)
bash scripts/package.sh         # -> prism-research-agent.zip

# 6. record the demo per docs/recording_runsheet.md, then submit zip + recording
```
