.PHONY: help install install-dev test clean build upload install-editable

help:
	@echo "COLA - Makefile Commands:"
	@echo ""
	@echo "  make install           - Install COLA package"
	@echo "  make install-dev       - Install COLA with dev dependencies"
	@echo "  make install-editable  - Install COLA in editable mode"
	@echo "  make test              - Run tests"
	@echo "  make lint              - Run linters"
	@echo "  make format            - Format code with black"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make build             - Build distribution packages"
	@echo "  make upload-test       - Upload to TestPyPI"
	@echo "  make upload            - Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-editable:
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 xai_cola counterfactual_explainer
	mypy xai_cola counterfactual_explainer

format:
	black xai_cola counterfactual_explainer scripts

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel
	python -m twine check dist/*

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

