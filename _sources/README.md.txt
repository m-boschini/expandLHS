[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15076931.svg)](https://doi.org/10.5281/zenodo.15076931)

## expandLHS

`expandLHS` is a Python module that implements a model-free expansion algorithm for a Latin Hypercube sample set.
The Latin Hypercube Sampling (LHS) is a stratified sampling technique that allows to generate $N$ near-random samples 
in the $P$-dimensional hypercube $[0, 1)^P$. It is a space-filling sampling strategy that ensures the one-dimensional 
projection property, i.e. the samples are uniformly distributed in each one-dimension projection. This module extends the usage of this technique by implementing an expansion algorithm. Starting from an initial LHS set of size $N$, `expandLHS` samples $M$ additional points in a LHS-like fashion trying to preserve the LHS properties at most.

This algorithm is introduced in
- *“LHS in LHS”: a new expansion strategy for Latin hypercube sampling in simulation design.* 
M. Boschini, D. Gerosa, A. Crespi, M. Falcone (to be published)

The code is distributed under version control at
- [github.com/m-boschini/expandLHS](https://github.com/m-boschini/expandLHS)

The documentation is available at
 - [m-boschini.github.io/expandLHS](https://m-boschini.github.io/expandLHS)

To install the code simply use

    pip install expandLHS

An example notebook can be found in the [documentation](https://m-boschini.github.io/expandLHS) together with a detailed description of the functions.

`expandLHS` is released under the MIT License. 
