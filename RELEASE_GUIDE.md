# Release Guide

This guide explains how to build and release COLA to PyPI.

## Prerequisites

1. Install build tools:
   ```bash
   pip install --upgrade pip setuptools wheel twine
   ```

2. Have PyPI credentials ready:
   - TestPyPI: https://test.pypi.org/
   - PyPI: https://pypi.org/

## Building the Package

### Option 1: Using Make (Recommended)

```bash
make build
```

This will:
- Clean previous builds
- Build source distribution (.tar.gz)
- Build wheel (.whl)
- Check package for errors

### Option 2: Using Python Directly

```bash
python setup.py sdist bdist_wheel
python -m twine check dist/*
```

### Option 3: Using Build Script

```bash
bash scripts/build_release.sh
```

## Version Management

1. Update version in `VERSION` file
2. Update version in `xai_cola/version.py`
3. Update version in `setup.py` and `pyproject.toml`
4. Update `CHANGELOG.md` with changes

Example:
```bash
echo "0.2.0" > VERSION
```

## Testing Before Release

### Test on TestPyPI First

```bash
# Build
make build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ xai-cola
```

### Test Installation Locally

```bash
# Install from local build
pip install dist/xai-cola-0.1.0.tar.gz

# Or in development mode
pip install -e .
```

## Releasing to PyPI

### Automated Release (Recommended)

```bash
bash scripts/publish_to_pypi.sh
```

### Manual Release

```bash
# Build package
make build

# Upload to PyPI
twine upload dist/*

# When prompted:
# Username: __token__
# Password: your-pypi-token
```

## Post-Release

1. Create a git tag:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. Create GitHub release from the tag

3. Update documentation if needed

## Checklist

Before each release:

- [ ] All tests pass
- [ ] Version updated in all files
- [ ] CHANGELOG.md updated
- [ ] Documentation updated
- [ ] README.md reviewed
- [ ] No debug code or print statements
- [ ] License and copyright updated
- [ ] Dependencies are up to date
- [ ] Example notebooks work correctly

## Troubleshooting

### Error: Package already exists on PyPI

Update version number before releasing.

### Error: Missing required files

Check `MANIFEST.in` includes all necessary files.

### Error: Import errors after installation

Ensure all dependencies are listed in `requirements.txt`.

## Semver Versioning

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new backward-compatible functionality
- **PATCH** version for backward-compatible bug fixes

Example: 1.2.3
- 1 = MAJOR
- 2 = MINOR
- 3 = PATCH

