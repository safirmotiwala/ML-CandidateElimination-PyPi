#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:02:36 2020

@author: safir
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="classic_CandidateElimination", 
    version="2.0.0",
    author="Safir Motiwala",
    author_email="safirmotiwala@gmail.com",
    description="Candidate-Elimination Algorithm is a Machine Learning Algorithm that builds the version space from Specific Hypothesis and General Hypothesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safir72347/ML-CandidateElimination-PyPi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)