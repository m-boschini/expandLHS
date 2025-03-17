from setuptools import setup, find_packages

# Read version from __version__.py
version = {}
with open("expandLHS/__version__.py") as f:
    exec(f.read(), version)

setup(
    name="expandLHS",
    version=version["__version__"],
    author="Matteo Boschini, Davide Gerosa",
    author_email="m.boschini1@campus.unimib.it, davide.gerosa@unimib.it",
    license="MIT",
    description="A Python package to expand a Latin Hypercube Sample",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/m-boschini/expandLHS",
    packages=find_packages(include=["expandLHS", "expandLHS.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "numba>=0.57"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
