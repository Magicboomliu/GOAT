#!/usr/bin/env python
"""
GOAT: Global Occlusion-Aware Transformer for Robust Stereo Matching
"""

from setuptools import setup, find_packages
import os

# Read the contents of README
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Read requirements
def read_requirements(fname):
    with open(fname) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='goat',
    version='1.0.0',
    author='Zihua Liu',
    author_email='',
    description='Global Occlusion-Aware Transformer for Robust Stereo Matching',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/Magicboomliu/GOAT',
    packages=find_packages(exclude=['scripts', 'tests', 'docs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'flake8>=3.8',
            'black>=20.8b1',
        ],
    },
    include_package_data=True,
    keywords='stereo matching, depth estimation, transformer, computer vision, deep learning',
    project_urls={
        'Paper': 'https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Global_Occlusion-Aware_Transformer_for_Robust_Stereo_Matching_WACV_2024_paper.pdf',
        'Project Page': 'http://www.ok.sc.e.titech.ac.jp/res/DeepSM/wacv2024.html',
    },
)

