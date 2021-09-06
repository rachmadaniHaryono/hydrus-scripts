#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hydrus-scripts setup script.
"""

from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = ["click", "pyyaml", "hydrus-api", "tqdm", "more_itertools"]

test_requirements = []

setup(
    author="rachmadani haryono",
    author_email="foreturiga@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="""personal hydrus scripts""",
    entry_points={
        "console_scripts": [
            "hydrus_scripts=hydrus_scripts:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    data_files=[(".", ["LICENSE", "HISTORY.md"])],
    keywords="hydrus_scripts",
    name="hydrus_scripts",
    py_modules=["hydrus_scripts"],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/rachmadaniharyono/hydrus_scripts",
    version="0.1.0",
    zip_safe=True,
)
