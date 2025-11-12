# Contributing to COLA

Thank you for your interest in contributing to COLA (COunterfactual explanations with Limited Actions)! We welcome contributions of all kinds: bug reports, documentation improvements, new features, and code optimizations.

This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/COLA.git
   cd COLA
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include Python version, OS, COLA version
- Provide minimal reproducible example
- Describe expected vs actual behavior

### 📖 Documentation
- Fix typos or clarify explanations
- Add examples or tutorials
- Improve docstrings
- Translate documentation

### ✨ New Features
- Open an issue first to discuss the feature
- Ensure it aligns with project goals
- Include tests and documentation

### 🔧 Code Quality
- Refactoring for better performance
- Adding type hints
- Improving test coverage

## Development Setup

### Prerequisites

- Python 3.7+ (Python 3.9+ recommended)
- Git
- pip or conda
- (Optional) C++ compiler for building some dependencies

### Setup Development Environment

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/your-username/COLA.git
cd COLA

# 3. Add upstream remote
git remote add upstream https://github.com/original-repo/COLA.git

# 4. Create virtual environment (highly recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install in editable mode with development dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/

# Check code formatting
black --check xai_cola/

# Run linter
flake8 xai_cola/
```

## Code Style

We follow PEP 8 style guidelines. Use the following tools:

- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking

Format your code before committing:
```bash
black xai_cola counterfactual_explainer
```

## Making Changes

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests to ensure everything works:
   ```bash
   make test
   ```

4. Format and lint your code:
   ```bash
   make format
   make lint
   ```

## Commit Messages

Write clear and descriptive commit messages:
- Use present tense ("Add feature" not "Added feature")
- Limit first line to 72 characters
- Reference issues/PRs when applicable

Example:
```
Add support for custom distance metrics in matching

This update allows users to specify custom distance functions
for the optimal transport matching policy.

Closes #123
```

## Pull Request Process

1. Update CHANGELOG.md with your changes
2. Update documentation if needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit PR with a clear description

## Testing

We use pytest for testing with high standards for code coverage.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=xai_cola --cov-report=html tests/

# Run specific test file
pytest tests/test_cola.py -v

# Run specific test function
pytest tests/test_cola.py::test_set_policy -v

# Run tests matching a keyword
pytest -k "matcher" tests/
```

### Writing Tests

When adding new features, please include tests:

```python
# tests/test_your_feature.py
import pytest
from xai_cola import COLA

def test_your_new_feature():
    """Test description"""
    # Arrange
    # ... setup code ...

    # Act
    result = your_function()

    # Assert
    assert result == expected_value
```

**Test Guidelines:**
- Aim for 80%+ code coverage for new code
- Test both success and failure cases
- Use descriptive test names
- Include docstrings explaining what is tested
- Mock external dependencies when appropriate

### Continuous Integration

All pull requests must pass CI tests before merging. The CI pipeline checks:
- All tests pass on Python 3.7, 3.8, 3.9, 3.10, 3.11
- Code coverage meets minimum threshold
- Code style follows Black formatting
- No linting errors from flake8
- Type hints pass mypy checks

## Documentation

Documentation is as important as code! When making changes:

### Code Documentation

```python
def your_function(param1: str, param2: int) -> bool:
    """Brief one-line description.

    More detailed explanation of what the function does,
    including any important details about behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> your_function("test", 42)
        True
    """
    pass
```

### User Documentation

When adding new features:
- Update [API_REFERENCE.md](API_REFERENCE.md) with new public APIs
- Add usage examples to [QUICKSTART.md](QUICKSTART.md) if applicable
- Create tutorial in `docs/tutorials/` for major features
- Update [README.md](README.md) if it affects quick start
- Add entry to [CHANGELOG.md](CHANGELOG.md)

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build Sphinx documentation
cd docs/
make html

# View in browser
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
# or
start _build/html/index.html  # Windows
```

## Code of Conduct

Please be respectful and professional in all interactions.

## Questions?

Feel free to reach out to the maintainers:
- Lei You (leiyo@dtu.dk)
- Lin Zhu (s232291@dtu.dk)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

