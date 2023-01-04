from setuptools import find_packages
from setuptools import setup
from version import __version__

# dependencies are unpinned to avoid conflicts with AOP API
install_dependencies = [
    "feature-engine==1.0.2",
    "joblib==1.0.1",
    "matplotlib==3.3.4",
    "numpy==1.20.1",
    "pandas==1.2.2",
    "scikit-learn==0.24.1",
    "scipy==1.6.0",
    "seaborn==0.11.1",
    "statsmodels==0.12.2",
]

setup(
    name="deploying_ml",
    description="",
    version=__version__,
    author="Qualogy Solutions",
    install_requires=install_dependencies,
    include_package_data=True,
)
