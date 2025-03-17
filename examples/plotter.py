import numpy as np
import matplotlib.pyplot as plt
from expandLHS import ExpandLHS


def _plot_optionals(
    N : int,
    M : int,
    *,
    fig : plt.Figure,
    ax : plt.Axes,
    labels : str,
    grid : bool,
    voids : np.ndarray | None,
    voids_color : str,
    voids_alpha : float,
    overlaps :  np.ndarray | None,
    overlaps_color : str,
    overlaps_alpha : float,
    x_label : str | None,
    y_label : str | None
    ):
    """
    Plot optional features of the LHS sample set.
    """

    if voids is not None:
        for i in range(N + M):
            if voids[i,0]:
                ax.axvspan(i/(N + M), (i + 1)/(N + M), color=voids_color, \
                           alpha=voids_alpha, label='voids')
            if voids[i,1]:
                ax.axhspan(i/(N + M), (i + 1)/(N + M), color=voids_color, \
                           alpha=voids_alpha, label='voids')
                
    if overlaps is not None:
        for i in range(N + M):
            if overlaps[i,0] > 1:
                ax.axvspan(i/(N + M), (i + 1)/(N + M), color=overlaps_color, \
                           alpha=overlaps_alpha, label='overlaps')
            if overlaps[i,1] > 1:
                ax.axhspan(i/(N + M), (i + 1)/(N + M), color=overlaps_color, \
                           alpha=overlaps_alpha, label='overlaps')
    
    if grid:
        ax.set_xticks(np.linspace(0, 1, N + M + 1), minor=False)
        ax.set_xticklabels([0]+['']*(N + M - 1)+[1])
        ax.set_yticks(np.linspace(0, 1,  N + M + 1), minor=False)
        ax.set_yticklabels([0]+['']*( N + M - 1)+[1])
        ax.grid(which='major', color='gray', linewidth=0.1)
        ax.tick_params(axis='both', which='both', length=0)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    
    if labels is not None:
        fig = ax.get_figure()
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = np.unique(labels)
        unique_handles = [handles[labels.index(l)] 
                          for l in unique_labels]
        fig.legend(unique_handles, unique_labels, loc='upper center', \
            ncol=unique_labels.shape[0], fontsize=12)
        
    return fig, ax
    
    

def plot(
    samples : np.ndarray | list[np.ndarray],
    M : int = 1,
    *,
    ax : plt.Axes = None,
    labels: str | list[str] | None = None,
    colors : str | list[str] = "red",
    markers : str | list[str] = "o",
    sizes : float | list[float] = 25.0,
    index : int = 0,
    grid : bool = True,
    voids : np.ndarray | None = None,
    voids_color : str = "grey",
    voids_alpha : float = 0.1,
    overlaps : np.ndarray | None = None,
    overlaps_color : str = "red",
    overlaps_alpha : float = 0.1,
    x_label : str | None = None,
    y_label : str | None = None
    ):
    """
    Plot LHS samples.
    
    Args:
    samples : np.ndarray | list[np.ndarray]
        LHS samples to plot. If a list of samples is provided, each sample 
        set will be plotted with the corresponding properties.
        
    M : int
        Number of samples to add to the initial LHS. Default is 1.
        
    ax : plt.Axes
        Axis to plot on. If None, a new figure is created.
        
    labels : str | list[str]
        Label for the samples. If None, no labels are shown.
        
    colors : str | list[str]
        Color of the scatter plot. Default is red.
        
    markers : str | list[str]
        Marker of the scatter plot. Default is "o".
        
    sizes : float | list[float]
        Size of the scatter plot. Default is 25.0.
        
    index : int
        Index of the sample set to use in voids and overlpas calculation. 
        If samples is a list, the default is the first sampple set (index = 0).
        
    grid : bool
        If True, show grid. Default is True.
        
    voids : np.ndarray | None
        Voids array. If None, voids are not shown. Default is None.
        
    voids_color : str
        Color of the voids. Unused if voids is None. Default is grey.
        
    voids_alpha : float
        Alpha of the voids. Unused if voids is None. Default is 0.1.
        
    overlaps : np.ndarray | None
        Overlaps array. If None, overlaps are not shown. Default is None.
        
    overlaps_color : str
        Color of the overlaps. Unused if overlaps is None. Default is red.
        
    overlaps_alpha : float  
        Alpha of the overlaps. Unused if overlaps is None. Default is 0.1.
    """
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        
    if isinstance(samples, list):        
        if len(samples) != len(colors):
            raise ValueError("Number of samples and colors must match.")
        
        if len(samples) != len(markers):
            raise ValueError("Number of samples and markers must match.")
        
        if len(samples) != len(sizes):
            raise ValueError("Number of samples and sizes must match.")
        
    else:
        samples = [samples]
        colors = [colors]
        markers = [markers]
        sizes = [sizes]
        
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        if len(samples) != len(labels):
            raise ValueError("Number of samples and labels must match.")
        
    for i, sample in enumerate(samples):
        P = sample.shape[1]
        if P != 2:
            raise ValueError("Only 2D samples are supported.")
        
        ax.scatter(sample[:, 0], sample[:, 1], \
                c=colors[i], s=sizes[i], marker=markers[i], \
                label=labels[i] if labels is not None else None)
    
        
    N = samples[index].shape[0]

    fig, ax = _plot_optionals(N, M, fig=fig, ax=ax, labels=labels, grid=grid, \
            voids=voids, voids_color=voids_color, voids_alpha=voids_alpha, \
            overlaps=overlaps, overlaps_color=overlaps_color, overlaps_alpha=overlaps_alpha,
            x_label=x_label, y_label=y_label)
        
    return fig, ax
    
    
    