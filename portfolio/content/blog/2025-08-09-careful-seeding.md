---
title: 'k-means++: The Advantages of Careful Seeding'
subtitle: 
author:
date: '2025-08-09'
draft: true
tags:
- Nonsmooth optimisation
- Clustering
---

## Introduction

This post will discuss and re-implement the `$k$`-means++ algorithm proposed by Arthur and Vassilvitskii (2007)[^1]. By understanding the algorithm, we may be able to integrate it with the clusterwise linear regression problem and measure its performance against existing methods.

If you have used the `scikit-learn` library to perform `$k$`-means clustering then likely you have used the `k-means++` algorithm to initialise the centroids. You get two options for initialisation:

1. Randomly select `$k$` points from the dataset as the initial centroids.
2. Use the `k-means++` algorithm to select the initial centroids.

For example:

```python
from sklearn.cluster import KMeans

option1 = KMeans(n_clusters=3, init='k-means++')
option2 = KMeans(n_clusters=3, init='random')
```

If you don't specificy the `init` method, you will use the `k-means++` algorithm by default.

We want to understand what this algorithm is doing from the ground up to build better intuition for it. In practice, however, we'd use scikit-learn's implementation since it's well tested and significantly more efficient than our own would be.

### Problem Formulation

In this discussion we will skip a lot detail around the `$k$`-means clustering problem and focus more on the `k-means++` algorithm and other related algorithms in the paper. 

I do want to state the problem formulation for the `$k$`-means clustering problem (exactly how the authors formulate it). This is only because it will be useful to have in mind when we discuss the `k-means++` algorithm.

Given `$k \in \mathbb{N}$` and a set of `$n$` points in `$\mathbb{R}^d$`, our objective is to choose `$k$` centers `$\{c_1, c_2, \ldots, c_k\} \subset \mathbb{R}^d$` to minimise the objective function `$\phi$`, which is the sum of squared distances between each point and its nearest center.

In code, we can write this objective function as:

```python
def phi(c, w, x, n, k):
    error = 0
    for i in range(n):
        for j in range(k):
            if w[i, j] == 1:
                error += np.sum((c[j] - x[i]) ** 2)
    return error / n
```

The assignment matrix `$w$` indicates which point is assigned to which center.

### The Lloyd Algorithm

Arthur and Vassilvitskii start by introducing the Lloyd algorithm which was proposed in 1957 but only officially published in 1982[^2]. 

The algorithm is a local search method that begins by selecting `$k$` starting points from the dataset at random. Each point from the dataset is then:

1. Assigned to the nearest `$k$` center, and then
2. Each `$k$` center is recomputed as the center of mass of all points assigned to it.
3. Do step 1 and 2 until the objective function converges.

The wikipedia page for the <a href="https://en.wikipedia.org/wiki/Lloyd%27s_algorithm" target="_blank" rel="noopener noreferrer">Lloyd algorithm</a> has a nice visualisation of the algorithm using a Voronoi diagram, which I have added below.

<img src="/images/2025-08-09-careful-seeding-1.gif" alt="Voronoi diagram of the Lloyd algorithm" height="300" style="display: block; margin: 0 auto; border-radius: 10px; box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.1);">

While I have heard of the Lloyd algorithm before, I have never really took the time to understand it. But now that I have, it is clear that the Lloyd algorithm is just the standard implementation of the `$k$`-means algorithm. 

*We will refer to the Lloyd algorithm as the `$k$`-means algorithm for the rest of this post.*

### Motivation for `$k$`-means++

Arthur and Vassilvitskii point out that while the `$k$`-means algorithm is very efficient, it can produce arbitrarily bad clusterings. More specifically, they state that the approximation ratio of `$\frac{\phi}{\phi_{OPT}}$` is unbounded even when `$n$` and `$k$` are fixed.

The key problem is due to random initialisation, since `$k$`-means is a local search algorithm, and greedily accepts improvements, it is possible to get stuck in a local minimum from globally bad starting points.

Due to the limitation of finding initial starting points, Arthur and Vassilvitskii propose a new algorithm, `$k$`-means++, which in fact still chooses `$k$` centers randomly but with very specific probabilities. 

In particular, they state:

>...we choose a point `$p$` as a center with probability proportional to `$p$`'s contribution to the overall potential.  Letting `$\phi$` denote the potential after choosing centers in this way, we show the following.
>
> Theorem 1.1. For any set of data points, `$E[\phi] \leq 8(\ln k+2)\phi_{OPT}$`

There are two parts here, which are preliminarily important to understand. First, choosing starting points with probability proportional to their contribution to the overall potential means the points that are chosen are likely to be spread out across our problem space. 

The second is the statement of the theorem which states that on average, the total potential is less than or equal to `$8(\ln k+2)$` times the optimal potential, which is a theoretical guarantee about how bad the clustering can be.



[^1]: Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (pp. 1027-1035). Society for Industrial and Applied Mathematics.

[^2]: Stuart P. Lloyd. Least squares quantization in pcm. IEEE Transactions on Information Theory, 28(2):129–136, 1982.