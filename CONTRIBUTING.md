# Contributing to COLA

Thank you for your interest in contributing to COLA! This document provides guidelines and instructions for contributing.

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

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/COLA.git
cd COLA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
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

We use pytest for testing. Run tests with:
```bash
make test
```

Or run specific tests:
```bash
pytest tests/test_specific.py -v
```

## Documentation

When adding new features:
- Update docstrings in code
- Add examples to demo notebooks
- Update README.md if needed
- Update API documentation

## Code of Conduct

Please be respectful and professional in all interactions.

## Questions?

Feel free to reach out to the maintainers:
- Lei You (leiyo@dtu.dk)
- Lin Zhu (s232291@dtu.dk)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

