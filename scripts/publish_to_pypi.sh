#!/bin/bash
# Publish COLA to PyPI

set -e

# Configuration
PACKAGE_NAME="xai-cola"
PYPI_REPOSITORY="pypi"  # or "testpypi" for testing

echo "Preparing to publish $PACKAGE_NAME..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Are you in the project root?"
    exit 1
fi

# Run build
echo "Building package..."
./scripts/build_release.sh

# Ask for confirmation
read -p "Are you sure you want to upload to $PYPI_REPOSITORY? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 1
fi

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload --repository $PYPI_REPOSITORY dist/*

echo "Upload complete! Check PyPI: https://pypi.org/project/$PACKAGE_NAME/"

