from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def read_version():
    with open(os.path.join('price_imputation_and_forecasting', '__init__.py')) as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

# Read requirements from the requirements.txt file
with open('requirements.txt', encoding='utf-8-sig') as f:
    required = f.read().splitlines()


setup(
    name='price_imputation_and_forecasting',
    version=read_version(),
    description='A library for imputing missing retail prices and forecasting future prices using deep learning pipelines.',
    author='Israel Kandleker',
    author_email='israel.kandleker@gmail.com',
    packages=find_packages(where='.'),  # Root directory of the package
    package_dir={'price_imputation_and_forecasting': 'price_imputation_and_forecasting'},  # Correct mapping
    include_package_data=True,
    install_requires=required,
    python_requires=">=3.8, <3.12",
    entry_points={
        "console_scripts": [
            "run_imputation_train_pipeline=scripts.run_imputation_train_pipeline:main",
            "run_forecasting_train_pipeline=scripts.run_forecasting_train_pipeline:main",
            "run_imputation_forecasting_predict_pipeline=scripts.run_imputation_forecasting_predict_pipeline:main",
        ]
    },
)