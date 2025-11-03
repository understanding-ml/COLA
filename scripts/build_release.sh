#!/bin/bash
# Build and release script for COLA

set -e

echo "Building COLA distribution..."

# Clean previous builds
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Install build dependencies
pip install --upgrade pip setuptools wheel twine

# Build the package
python setup.py sdist bdist_wheel

# Check the built package
echo "Checking package..."
python -m twine check dist/*

# Generate distribution archives
echo "Distribution archives created in dist/"

echo "Build complete! Upload to PyPI with:"
echo "twine upload dist/*"

