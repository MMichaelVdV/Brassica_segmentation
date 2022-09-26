#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'pippa-cnn-segmentation'
DESCRIPTION = 'Image analysis using deep learning.'
URL = 'https://gitlab.psb.ugent.be/Brassica/script_soup/2018-09-24_brassica_segmentation'
EMAIL = 'sam.demeyer@psb.ugent.be'
AUTHOR = 'Sam De Meyer <samey>'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    'hooloovoo @ git+https://gitlab.psb.ugent.be/Utilities/hooloovoo.git@master',
    'matplotlib',
    'numpy',
    'pandas',
    'pillow',
    'scikit-image',
    'scipy',
    'torch >= 1.1',
    'torchvision',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    here = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "applications"]),
    entry_points={
        'console_scripts': ['pippa_cnn_segmentation=pippa_cnn_segmentation.libs.command_line:main'],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
