import numpy as np
from scipy.stats.qmc import discrepancy, geometric_discrepancy
from numba import jit, types as ntypes
from typing import Optional, Union, Tuple, Literal


@jit(nopython=True)
def _regridding(
    N : ntypes.u8,
    P : ntypes.u8,
    samples : np.ndarray,
    M : ntypes.u8
    ) -> np.ndarray:
    """
    Numba JITed code for ExpandLHS.regridding().
    """
    
    intervals = np.floor(samples * (N + M)).astype(ntypes.u8)
    voids = np.ones((N + M, P), dtype=ntypes.b1)
    
    for j in range(P):
        voids[intervals[:, j], j] = 0

    return voids


@jit(nopython=True)
def _count_samples(
    N : ntypes.u8,
    P : ntypes.u8,
    samples : np.ndarray,
    M : ntypes.u8
    ) -> np.ndarray:
    """
    Numba JITed code for ExpandLHS.count_samples().
    """
    
    intervals = np.floor(samples * (N + M)).astype(ntypes.u8)
    population = np.zeros((N + M, P), dtype=ntypes.u8)
    
    for i in range(N):
        for j in range(P):
            population[intervals[i, j], j] += 1
        
    return population


@jit(nopython=True)
def _degree(
    N : ntypes.u8,
    P : ntypes.u8,
    samples : np.ndarray,
    M : ntypes.u8
    ) -> ntypes.f8:
    """
    Numba JITed code for ExpandLHS.degree().
    """
    
    voids = _regridding(N, P, samples, M)
    return 1 - (np.sum(voids) - M * P) / ((N + M) * P)


@jit(nopython=True)
def _optimal_expansion(
    N : ntypes.u8,
    P : ntypes.u8,
    samples : np.ndarray,
    radius : Tuple[ntypes.u8, ntypes.u8]
    ) -> list[ntypes.u8, ntypes.f8]:
    """
    Numba JITed code for ExpandLHS.optimal_expansion().
    """
        
    scaling_up = [(_degree(N, P, samples, radius[0]), radius[0])]
    for m in range(radius[0]+1, radius[1]+1):
        scaling_up += [(_degree(N, P, samples, m), m)]
        
    scaling_up = [(_degree(N, P, samples, 0), 0)] + \
        sorted(scaling_up, reverse=True)
    scaling_up = [(tmp[1], tmp[0]) for tmp in scaling_up]
    
    return scaling_up


@jit(nopython=True)
def _LHSinLHS_sampling(
    N : ntypes.u8,
    P : ntypes.u8,
    M : ntypes.u8, 
    voids : np.ndarray,
    rng : np.random.Generator
    ) -> np.ndarray:
    """
    Numba JITed code for ExpandLHS.LHSinLHS_sampling().
    """
    
    new_samples = np.empty((M, P))
    intervals = np.arange(0, 1, 1 / (N + M))
    
    for j in range(P):
        # TODO: change np.random.choice in rng.choice when Numba will support it
        l_bounds = np.random.choice(intervals[voids[:,j]], M, replace=False)
        u_bounds = l_bounds + 1 / (N + M)
    
        new_samples[:,j] = [rng.uniform(l_bound, u_bound) \
            for l_bound, u_bound in zip(l_bounds, u_bounds)]
    
    return new_samples
    


class ExpandLHS:
    """
    Class of the Latin Hypercube Sampling (LHS) Expansion algorithm.
    
    The Latin Hypercube Sampling is a stratified sampling method that
    allows to generate N near-random samples in the P-dimensional hypercube
    [0, 1)^P. It is a space-filling sampling strategy that ensures the 
    one-dimensional projection property, i.e. the samples are uniformly
    distributed in each one-dimension projection.
    
    Given an initial LHS set of size N defined in the P-dimensional hypercube 
    [0, 1)^P, this class expands the set adding new M samples.
    This procedure requires a new partition of the interval [0,1) in each 
    dimension P that may result in loosing the LHS projection property. The case
    M = k * N, where k is a natural number, always preserves the LHS projection 
    property. 
    This code allows to compute a new metric D, with 0 < D <= 1, corresponding 
    to the degree-of-LHS of a samples set, where D = 1 corresponds to a perfect 
    LHS. This allows to quantify the impact of an epansion on the initial set.
    It is worth noticing that the degree does not depend on the sampling 
    strategy, but only on the initial set and the new partition of the intervals.
    
    Attributes
    ----------
    N : int
        Number of samples.
        
    P : int
        Number of dimensions.
        
    samples : array_like (N, P)
        Initial Latin Hypercube sample set.
   
    
    Methods
    -------
    regridding
        Compute the new partition of the interval [0,1).
        
    count_samples 
        Compute the number of samples in each interval and dimension.
        
    degree 
        Compute the degree-of-LHS of a samples set.
        
    optimal_expansion 
        Find the optimal expansion size M maximizing the final degree.
        
    LHSinLHS_sampling 
        Sample M additional samples filling M empty intervals created by 
        the new partition in each dimension P.
        
    LHSinLHS_optimized
        Sample M additional samples and optimize the their spatial distribution. 
        
    __call__ 
        Expand a given sample set adding additional M samples.
    """
    
    def __init__(
        self, 
        samples : np.ndarray
        ) -> None:
        """
        Initialize the LHSExpansion class.

        Args:
            samples : array_like (N, P)
                Latin Hypercube sample set with N samples in P dimensions. 
                
        """
        
        self.N = samples.shape[0]
        self.P = samples.shape[1]
        self.samples = samples
        
        self.rng = np.random.default_rng()
            
            
        
    

    def regridding(
        self,
        M : int = 1,
        *,
        samples : Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        Regrid the Latin Hypercube samples by adding M new intervals in 
        each dimension.

        Args:
            M : int (optional) 
                Number of new intervals to add. Defaults to 1.
            
            samples : array_like (optional)
                If given, the regridding will be performed on this set and not
                on the default one.

        Returns:
            voids : numpy.ndarray (N + M, P)
                A boolean array indicating the empty intervals. 
                In each dimension the number of voids is >= M, as adding M 
                intervals may cause two samples to fall in the same bin, 
                thus leaving a permanent void.
        """
        
        if samples is None:
            N = self.N
            P = self.P
            samples = self.samples
            
        else:
            N = samples.shape[0]
            P = samples.shape[1]
        
        voids = _regridding(N, P, samples, M)

        return voids
    
    
    def count_samples(
        self,
        M : int = 1,
        *,
        samples : Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        Count the number of samples in each of the (N + M) x P intervals.

        Args:
            M : int (optional)
                Number of new intervals to add. Defaults to 1.
                
            samples : arrray_like (optional)
                If given, the counting of samples will be performed on this set 
                and not on the default one.

        Returns:
            population : numpy.ndarray (N + M, P) 
                An array indicating the number of samples in each interval.
                Adding M intervals may cause two samples to fall in the same 
                bin, thus leaving a permanent void.
        """
        
        if samples is None:
            N = self.N
            P = self.P
            samples = self.samples
            
        else:
            N = samples.shape[0]
            P = samples.shape[1]
        
        population = _count_samples(N, P, samples, M)
            
        return population
    
    
    def degree(
        self,
        M : int = 1
        ) -> float:
        """
        Compute the degree-of-LHS of the current sample set when expanded to 
        size N + M, assuming M new samples will be generated.
        If M = 0, compute the degree of the initial set.

        Args:
            M : int (optional)
                Number of new intervals to add. Defaults to 1.

        Returns:
            degree : float
                Degree (D) of the Latin Hypercube Sampling, with 0 < D <= 1.
        """
        
        lhs_degree = _degree(self.N, self.P, self.samples, M)
        
        return lhs_degree
    
    
    def optimal_expansion(
        self,
        radius : Union[int, Tuple[int, int]],
        verbose : bool = False
        ) -> list[int, float]:
        """
        Find the optimal expansion size ---the expansion that has 
        the higher degree-of-LHS in a given range.

        Args:
            radius : int | (int, int)
                Range of values to consider for the expansion. 
                If a single value is provided, it is interpreted as 
                [1, upper bound] 
                If a tuple is provided, it is interpreted as 
                [lower bound, upper bound].
                Both the lower and the upper bounds are included.
                
            verbose : bool
                If False return the expansion size with the highest degree, 
                if True returns all the expansion sizes within radius.
                Defaults to False.
                
        Returns:
            expansions size, expansion degree : [(int, float),...]
                List of tuples of kind (expansion size, expansion degree),
                ordered by decreasing expansion degree. 
                The tuple with expansion size equal 0 is the degree of the 
                current set (=1 for a perfect Latin Hypercube).
        """
        
        if not isinstance(radius, tuple):
            radius = (1, radius)
        
        expansions = _optimal_expansion(self.N, self.P, self.samples, radius)
        
        if verbose:
            return expansions
        
        else:
            # expansions[0] is always the degree of the initial set
            return expansions[1]
    
    
    def LHSinLHS_sampling(
        self,
        M : int, 
        voids : np.ndarray,
        ) -> np.ndarray:
        """
        Generate new samples within the voids of the current sample set 
        preserving as much as possible the properties of a Latin Hypercube.

        Args:
            M : int 
                Number of new samples to generate.
                
            voids : array_like (N + M, P) 
                Boolean array indicating the empty intervals.

        Returns:
            new samples : numpy.ndarray (M, P)
                New samples generated within the voids.
        """
        
        new_samples = _LHSinLHS_sampling(self.N, self.P, M, voids, self.rng)
        
        return new_samples
    
    
    def LHSinLHS_optimized(
        self,
        M : int, 
        voids : np.ndarray,
        criterion : str,
        trials : int,
        tol : float,
        ) -> np.ndarray:
        """
        Generate new samples within the voids of the current sample set 
        preserving as much as possible the properties of a Latin Hypercube. 
        The samples spatial distribution is optimize in order to achieve the 
        lowest centered discrepancy or the higher geometric discrepancy within
        a number of samplings equal to trials.

        Args:
            M : int
                Number of new samples to generate.
                
            voids : array_like (N + M, P) 
                Boolean array indicating the empty intervals.
                
            criterion : str 
                Optimization strategy.
                
            trials : int 
                Number of expansions to sample. 
                
            tol : float
                Tolerance for the optimization.

        Returns:
            opt_samples : numpy.ndarray (M, P) 
                New samples generated within the voids and optimized with the 
                given criterion.
        """
        
        opt_mode = {
            'discrepancy' : discrepancy,
            'geometric_discrepancy' : geometric_discrepancy
            }
        
        opt_samples = _LHSinLHS_sampling(self.N, self.P, M, voids, self.rng)
        opt_criterion = opt_mode[criterion](
                np.concatenate([self.samples, opt_samples], axis=0))
        
        for _ in range(trials):
            tmp_samples = _LHSinLHS_sampling(self.N, self.P, M, voids, self.rng)
            tmp_criterion = opt_mode[criterion](
                np.concatenate([self.samples, tmp_samples], axis=0))
            
            if np.abs(tmp_criterion - opt_criterion) < tol:
                break
            
            elif criterion == 'discrepancy' and tmp_criterion < opt_criterion:
                opt_samples = tmp_samples
                opt_criterion = tmp_criterion 
                
            elif criterion == 'geometric_discrepancy' and \
                tmp_criterion > opt_criterion:
                opt_samples = tmp_samples
                opt_criterion = tmp_criterion
        
        return opt_samples

            
    def __call__(
        self, 
        M : int = 1,
        *,
        seed : Optional[int] = None,
        optimize : Optional[Literal['discrepancy', 'geometric_discrepancy']] = None,
        trials : int = 1000,
        tol : float = 1e-4
        ) -> np.ndarray:
        """
        Build the Latin Hypercube expansion of size M.
        
        The optional argument 'optimize' allows to find an optimal set of new 
        samples with lower centered discrepancy (space filling metric) 
        or higher geometric discrepancy (pairwise minimum distance).
        Both these metrics are implemented in Scipy.stats.qmc.
        The code generates a number of possible expansion equal to 'trials' and 
        returns the best set with respect to the choosen metric within a certain 
        absolute tolerance defined by 'tol'. 
        It is not garanteed that the final sample will be the absolute minimum 
        of the discrepancy or the absolute maximum of the geometric 
        discrepancy for the given sample. The optimization through geometric
        discrepancy could fail if the minimum pairwise distance is set by the 
        initial set.
        

        Args:
            M : int (optional)
                Number of new samples to generate. Defaults to 1.
            
            seed : int | None (optional)
                Seed for the random number generator. Defaults to None.
                
            optimize : str | None (optional)
                Optimization criterion, the available options are 'discrepancy'
                and 'geometric_discrepancy'. Defaults to None.
                
            trials : int (optional)
                Number of trials for the optimal expansion set. Defaults to 1000.
                
            tol : float (optional)
                Tolerance for the optimization. Defaults to 1e-4.

        Returns:
            expansion : numpy.ndarray (N + M, P)
                Expanded Latin Hypercube sample set.
        """
        
        if M == 0:
            return self.samples
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        voids = self.regridding(M)
        
        if optimize is None:
            new_samples = self.LHSinLHS_sampling(M, voids)
        else:
            new_samples = self.LHSinLHS_optimized(M, voids, optimize, trials, tol)
                    
        return np.concatenate([self.samples, new_samples], axis=0)
