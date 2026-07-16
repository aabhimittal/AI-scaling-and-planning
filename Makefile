.PHONY: help install dev test lint fmt demo serve docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

dev: ## Install with dev/test dependencies
	pip install -e ".[dev]"

test: ## Run the test suite
	pytest

lint: ## Lint with ruff
	ruff check src tests

fmt: ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

demo: ## End-to-end demo: generate -> train -> simulate (+ plot)
	predictive-scaling generate --out artifacts/load.csv --days 45
	predictive-scaling train --data artifacts/load.csv --out artifacts/model.joblib --report artifacts/report.json
	predictive-scaling simulate --data artifacts/load.csv --out artifacts/sim.csv --plot artifacts/sim.png --eval-steps 2016

serve: ## Run the API locally (http://localhost:8000/docs)
	predictive-scaling serve --reload

docker: ## Build the container image
	docker build -t predictive-scaling:latest .

clean: ## Remove build artifacts and caches
	rm -rf artifacts build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
