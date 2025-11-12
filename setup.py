#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README.md file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements = []
if Path("requirements.txt").exists():
    with open("requirements.txt", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() 
            and not line.startswith("#") 
            and not line.startswith("torch")  # Exclude torch as it's optional
        ]

setup(
    name="xai-cola",
    version="0.1.5",
    description="COunterfactual explanations with Limited Actions (COLA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lin Zhu, Lei You",
    author_email="s232291@student.dtu.dk",
    url="https://github.com/understanding-ml/COLA",
    license="MIT",
    python_requires=">=3.9",
    install_requires=requirements,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "examples", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="explainable AI, XAI, counterfactual explanations, machine learning, interpretability",
    project_urls={
        "Bug Reports": "https://github.com/understanding-ml/COLA/issues",
        "Source": "https://github.com/understanding-ml/COLA",
        "Documentation": "https://cola.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)

