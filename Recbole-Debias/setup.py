from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = []

setup_requires = []

extras_require = {}

classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "RecBole-Debias is a toolkit built upon RecBole for reproducing and developing debiased recommendation algorithms."
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="recbole-debias",
    version="1.0.0",
    description="A unified, comprehensive and efficient recommendation library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JingsenZhang/Recbole-Debias/tree/master",
    author="JingsenZhang",
    packages=[package for package in find_packages() if package.startswith("recbole_debias")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
