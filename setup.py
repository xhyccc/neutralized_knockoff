from setuptools import setup, find_packages

setup(
    name="knockoff_neutralized",
    version="0.1.0",
    description="A robust quantitative trading strategy using conditional knockoff filters and factor neutralization",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "cvxpy>=1.2.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
