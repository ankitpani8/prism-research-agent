# Multi-stage build running the FastAPI surface. Filled in Phase 7.
# (slim base, cached deps layer, non-root user, mem/CPU limits set in compose)
FROM python:3.11-slim AS base
# 🚧 Phase 7
