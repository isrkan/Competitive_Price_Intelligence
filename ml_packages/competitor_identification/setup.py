from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def read_version():
    with open(os.path.join('competitor_identification', '__init__.py')) as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

# Read requirements from the requirements.txt file
with open('requirements.txt', encoding='utf-8-sig') as f:
    required = f.read().splitlines()


setup(
    name='competitor_identification',
    version=read_version(),
    description='A library for identifying competitors in retail markets using dimensionality reduction and clustering methods.',
    author='Israel Kandleker',
    author_email='israel.kandleker@gmail.com',
    packages=find_packages(where='.'),  # Root directory of the package
    package_dir={'competitor_identification': 'competitor_identification'},  # Correct mapping
    include_package_data=True,
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'run_prediction_pipeline=competitor_identification.scripts.run_prediction_pipeline:main',
        ]
    },
)