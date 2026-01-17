# Superpixel Methods Comparison

Comparative study and implementation of superpixel extraction methods: SLIC and Hierarchical Segmentation via Structural Information Theory.

## Overview

This project implements and compares two superpixel segmentation methods:
- SLIC (Simple Linear Iterative Clustering)
- Hierarchical Superpixel Segmentation via Structural Information Theory

The goal is to provide both qualitative and quantitative evaluation of these methods.

## Motivation

Despite the dominance of deep learning in computer vision, superpixel methods remain relevant for several reasons:

- Computational efficiency through reduced number of primitives
- Interpretable mid-level representations
- Useful for weakly-supervised and few-shot learning
- Integration as preprocessing for neural networks
- Essential for resource-constrained applications and satellite imagery

## Methods

### Method 1: SLIC

Paper: [SLIC Superpixels Compared to State-of-the-art Superpixel Methods](https://ieeexplore.ieee.org/document/6205760)

Key features:
- K-means clustering in 5D space (Lab color + XY coordinates)
- Distance metric combining color and spatial proximity
- Efficient implementation with limited search space

### Method 2: Hierarchical Superpixel Segmentation

Paper: [Hierarchical Superpixel Segmentation via Structural Information Theory](https://arxiv.org/pdf/2501.07069)

Key features:
- Information-theoretic approach
- Multi-scale hierarchical representation
- Optimized for satellite and remote sensing imagery

## Usage

TODO

## Evaluation

### Quantitative Metrics

TODO
- Boundary Recall (BR)
- Under-segmentation Error (UE)
- Achievable Segmentation Accuracy (ASA)
- Compactness
- Computation Time

### Qualitative Evaluation

TODO
- Human perception study with 10 testers
- Visual comparison
- Comparison with IPOL implementation: https://www.ipol.im/pub/art/2022/373/

## References

1. Achanta, R., et al. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. IEEE TPAMI.
   - https://ieeexplore.ieee.org/document/6205760

2. Hierarchical Superpixel Segmentation via Structural Information Theory (2025).
   - https://arxiv.org/pdf/2501.07069

3. Application to satellite images.
   - https://arxiv.org/html/2411.17922v2

4. IPOL SLIC Implementation.
   - https://www.ipol.im/pub/art/2022/373/

## License

MIT License

## Contact

For questions, please open an issue.