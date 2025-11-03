#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README.rst file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text(encoding="utf-8")

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
    version="0.1.0",
    description="COunterfactual explanations with Limited Actions (COLA)",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Lei You, Yijun Bian, Lin Zhu",
    author_email="leiyo@dtu.dk",
    url="https://github.com/your-repo/COLA",
    license="MIT",
    python_requires=">=3.8",
    install_requires=requirements,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "examples", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="explainable AI, XAI, counterfactual explanations, machine learning, interpretability",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/COLA/issues",
        "Source": "https://github.com/your-repo/COLA",
        "Documentation": "https://cola.readthedocs.io/",
        "Paper": "https://arxiv.org/pdf/2410.05419",
    },
    include_package_data=True,
    zip_safe=False,
)

