# Legal Chatbot Makefile

.PHONY: help setup install test lint format run clean docker-build docker-run

help: ## Show this help message
	@echo "Legal Chatbot - Available Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the development environment
	@echo "Setting up development environment..."
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt
	. venv/bin/activate && pre-commit install
	@echo "Environment setup complete!"

install: ## Install dependencies
	. venv/bin/activate && pip install -r requirements.txt

install-dev: ## Install development dependencies
	. venv/bin/activate && pip install -r requirements-dev.txt

test: ## Run tests
	. venv/bin/activate && pytest tests/ -v --cov=app --cov-report=html

test-unit: ## Run unit tests only
	. venv/bin/activate && pytest tests/unit/ -v

test-integration: ## Run integration tests only
	. venv/bin/activate && pytest tests/integration/ -v

lint: ## Run linting
	. venv/bin/activate && flake8 app/ tests/
	. venv/bin/activate && black --check app/ tests/
	. venv/bin/activate && isort --check-only app/ tests/

format: ## Format code
	. venv/bin/activate && black app/ tests/
	. venv/bin/activate && isort app/ tests/

run: ## Run the application
	. venv/bin/activate && uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Run Streamlit frontend
	. venv/bin/activate && streamlit run frontend/streamlit/app.py --server.port 8501

run-all: ## Run both API and frontend
	@echo "Starting API server..."
	. venv/bin/activate && uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting frontend..."
	. venv/bin/activate && streamlit run frontend/streamlit/app.py --server.port 8501

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage

docker-build: ## Build Docker image
	docker build -t legal-chatbot .

docker-run: ## Run with Docker Compose
	docker-compose up --build

docker-stop: ## Stop Docker containers
	docker-compose down

notebook: ## Start Jupyter notebook
	. venv/bin/activate && jupyter notebook notebooks/

notebook-lab: ## Start Jupyter Lab
	. venv/bin/activate && jupyter lab notebooks/

docs: ## Generate documentation
	. venv/bin/activate && mkdocs serve

security-scan: ## Run security scans
	. venv/bin/activate && safety check
	. venv/bin/activate && bandit -r app/

data-download: ## Download required datasets
	@echo "Downloading CUAD dataset..."
	python scripts/setup/download_cuad.py
	@echo "Downloading UK legislation..."
	python scripts/setup/download_legislation.py

setup-db: ## Set up database
	. venv/bin/activate && python scripts/setup/setup_database.py

migrate: ## Run database migrations
	. venv/bin/activate && alembic upgrade head

seed-data: ## Seed database with initial data
	. venv/bin/activate && python scripts/setup/seed_data.py
