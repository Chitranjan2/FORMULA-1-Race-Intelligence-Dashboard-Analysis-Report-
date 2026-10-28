from setuptools import find_packages, setup

setup(
    name="f1-race-intelligence-analytics",
    version="2.0.0",
    author="Dhyey Teraiya",
    author_email="dhyeyteraiya@gmail.com",
    description=(
        "Real-data Formula 1 analytics: fuel- and tyre-corrected pace, measured degradation, "
        "pit-loss strategy simulation and walk-forward podium prediction"
    ),
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.0",
        "matplotlib>=3.8.0",
        "streamlit>=1.32.0",
        "pyarrow>=14.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
)
