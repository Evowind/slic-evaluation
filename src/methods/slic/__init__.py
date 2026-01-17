"""
Module SLIC (Simple Linear Iterative Clustering)
"""
from .slic_original import SLIC
from .utils import (
    visualize_superpixels,
    compute_superpixel_statistics,
    get_average_colors,
    compare_parameters,
    analyze_compactness
)

__all__ = [
    'SLIC',
    'visualize_superpixels',
    'compute_superpixel_statistics',
    'get_average_colors',
    'compare_parameters',
    'analyze_compactness'
]
