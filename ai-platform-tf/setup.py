
# python3
# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==2.1.0',
    'numpy==1.18.0',
    'pandas==1.2.1',
    'scipy==1.4.1',
    'scikit-learn==0.22',
    'google-cloud-storage==1.23.0',
    'xgboost==1.3.3',
    'cloudml-hypertune',
    ]
 
setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Trainer package for XGBoost Task'
)
