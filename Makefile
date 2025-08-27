# Makefile for Subgroup Fairness Experiments
# Provides convenient commands for testing, development, and deployment

.PHONY: help install test test-unit test-integration test-performance test-fast test-coverage clean lint format setup-dev

# Default target
help:
	@echo "Subgroup Fairness Research - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make install          Install all dependencies"
	@echo "  make setup-dev        Setup development environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"  
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-performance Run performance benchmarks"
	@echo "  make test-fast       Run fast tests (skip slow ones)"
	@echo "  make test-coverage   Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run code linting"
	@echo "  make format          Format code with black/isort"
	@echo "  make clean           Clean temporary files"
	@echo ""
	@echo "Experiments:"
	@echo "  make run-experiments Run all experiments"
	@echo "  make run-exp-1       Run accuracy-fairness experiment"
	@echo ""

# Installation and setup
install:
	@echo "📦 Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements_test.txt
	@echo "✅ Dependencies installed"

setup-dev: install
	@echo "🔧 Setting up development environment..."
	pip install pre-commit black isort flake8
	pre-commit install
	@echo "✅ Development environment ready"

# Testing commands
test:
	@echo "🧪 Running all tests..."
	python run_tests.py --verbose

test-unit:
	@echo "🔬 Running unit tests..."
	python run_tests.py --unit --verbose

test-integration:
	@echo "🔗 Running integration tests..."
	python run_tests.py --integration --verbose

test-performance:
	@echo "⚡ Running performance benchmarks..."
	python run_tests.py --performance --verbose

test-fast:
	@echo "🏃 Running fast tests..."
	python run_tests.py --fast --verbose

test-coverage:
	@echo "📊 Running tests with coverage..."
	python run_tests.py --coverage --html-cov --verbose
	@echo "📋 Coverage report: htmlcov/index.html"

# Code quality
lint:
	@echo "🔍 Running code linting..."
	flake8 scripts/ tests/ --count --statistics --show-source
	@echo "✅ Linting complete"

format:
	@echo "🎨 Formatting code..."
	black scripts/ tests/ *.py --line-length 100
	isort scripts/ tests/ *.py --profile black
	@echo "✅ Code formatted"

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete"

# Experiment running
run-experiments:
	@echo "🔬 Running all experiments..."
	python scripts/experiments/main_experiment_pipeline.py --experiment exp_1_accuracy_fairness --output-dir results
	python scripts/experiments/main_experiment_pipeline.py --experiment exp_2_computational --output-dir results
	python scripts/experiments/main_experiment_pipeline.py --experiment exp_3_partial_complete --output-dir results
	@echo "✅ All experiments complete"

run-exp-1:
	@echo "📈 Running accuracy-fairness trade-off experiment..."
	python scripts/experiments/main_experiment_pipeline.py --experiment exp_1_accuracy_fairness --output-dir results --log-level INFO
	@echo "✅ Experiment 1 complete"

run-exp-2:
	@echo "⚡ Running computational efficiency experiment..."
	python scripts/experiments/main_experiment_pipeline.py --experiment exp_2_computational --output-dir results --log-level INFO
	@echo "✅ Experiment 2 complete"

# CI/CD simulation
ci-test:
	@echo "🔄 Running CI test suite..."
	python run_tests.py --coverage --exitfirst --quiet
	@echo "✅ CI tests passed"

# Development helpers
dev-test:
	@echo "👨‍💻 Running development test suite..."
	python run_tests.py --unit --fast --verbose

check: lint test-fast
	@echo "✅ Pre-commit checks passed"

# Docker commands (if using containerization)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t subgroup-fairness .
	@echo "✅ Docker image built"

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker run --rm subgroup-fairness make test
	@echo "✅ Docker tests complete"

# Benchmarking
benchmark:
	@echo "📊 Running performance benchmarks..."
	python run_tests.py --performance --benchmark-only --verbose
	@echo "✅ Benchmarks complete"

# Documentation generation (if using Sphinx or similar)
docs:
	@echo "📚 Generating documentation..."
	# Add documentation generation commands here
	@echo "✅ Documentation generated"

# Requirements management
update-deps:
	@echo "📦 Updating dependencies..."
	pip list --outdated
	@echo "Review and update requirements.txt manually"

freeze-deps:
	@echo "🧊 Freezing current dependencies..."
	pip freeze > requirements_frozen.txt
	@echo "✅ Dependencies frozen to requirements_frozen.txt"