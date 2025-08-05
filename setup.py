#!/usr/bin/env python3
"""
Setup script para instalação do projeto QAOA Turbinas Eólicas
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("CLAUDE.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qaoa-turbinas-eolicas",
    version="1.0.0",
    author="Marcos",
    description="Algoritmo QAOA para otimização de posicionamento de turbinas eólicas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="quantum computing, QAOA, wind turbines, optimization, qiskit",
    entry_points={
        "console_scripts": [
            "qaoa-turbinas=qaoa_turbinas:main",
        ],
    },
)