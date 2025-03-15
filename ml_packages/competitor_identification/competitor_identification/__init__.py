# Import modules or functions to be directly accessible when importing the package
from .data import load_data, impute_data, preprocess_data, scale_data
from .dimensionality_reduction import dimensionality_reduction
from .clustering import clustering
from .pipeline import run_pipeline
from .config import config

# Define package version
__version__ = '0.1.0'