from setuptools import setup, find_packages

setup(
    name="bcishield",
    version="0.1.0",
    description="BCIShield: Adversarial Robustness Research for EEG-based BCI Systems",
    author="Research Engineer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "mne>=1.4.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0",
        "pytest>=7.3.0",
    ],
)
