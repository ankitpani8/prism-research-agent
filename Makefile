.PHONY: install dev run demo api test lint typecheck eval docker clean diagram

install:        ## install runtime deps
	pip install -r requirements.txt

dev:            ## install runtime + dev/test deps
	pip install -r requirements-dev.txt

run:            ## launch the Streamlit demo UI
	streamlit run streamlit_app.py

demo:           ## run the CLI trace (set Q="your question")
	python run_demo.py "$(Q)" --image data/sample_dashboard.png

api:            ## launch the FastAPI production surface
	uvicorn api.app:app --reload --port 8000

test:           ## run the full test suite (the "test rail")
	pytest -v

lint:           ## ruff lint
	ruff check .

typecheck:      ## mypy on core/
	mypy core

eval:           ## run the eval harness + regression demo
	python eval/run_eval.py

diagram:        ## export the Mermaid diagram from the compiled graph
	python -c "from core.graph import export_diagram; export_diagram()"

docker:         ## build + run the containerized stack
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
