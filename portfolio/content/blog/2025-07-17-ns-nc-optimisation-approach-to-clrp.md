---
title: Nonsmooth Nonconvex Optimization Approach to Clusterwise Linear Regression Problems
subtitle: 
author:
date: '2025-07-17'
draft: true
tags:
- Clustering
- Nonsmooth optimisation
- Clusterwise Linear Regression
---

This post builds on my previous <a href="/blog/2025-07-11-spath-cluster-wise-regression" target="_blank" rel="noopener noreferrer">post</a> where I discussed and re-implemented the Späth algorithm[^2]. In this post we will look at the work done by Bagirov et al. (2013)[^1] and their proposed incremental approach to solving the clusterwise linear regression problem. 

Like my other posts, we will try to understand the math using programming examples and because of this we will try to explicity write the code in pure Python instead of using NumPy functions. However, if it makes sense to use NumPy for some of the more complex operations we will do so (especially for when saving real estate).

**Note:** Since we have covered the preliminaries in my previous posts, we will not repeat them here. If you are not familiar with the <a href="/blog/2025-07-11-spath-cluster-wise-regression" target="_blank" rel="noopener noreferrer">Späth algorithm</a> or <a href="/blog/2025-06-27-modified-global-k-means" target="_blank" rel="noopener noreferrer">modified global k-means</a> algorithm, please refer to my previous posts for more details.

## Problem Formulation

### Dataset

Given a dataset `$A = \{(a^i, b_i) \in \mathbb{R}^n \times \mathbb{R}: 1, ... , m\}$`, the goal of clusterwise regression is to find not only an optimal partition of `$k$` clusters but also the optimal regression coefficients `$\{x_j, y_j\}, j=1,...,k$` within clusters in order to minimise the overall fit.

Each data point has features `$a^i$` (which can be `$n$`-dimensional) and `$b_i$` is a single scalar value which we want to predict. 

To show this programmatically, we can use the following example:

```python
m = 25
n = 1

a1 = np.random.uniform(0.6, 1.0, m)
b1 = 3 * a1 + 2 + np.random.normal(0, 0.1, m)

A = np.column_stack([a1, b1])
```

However, this only generates a single cluster of data points. In our example we will solve `$k=4$` which means we will need to generate 3 more clusters of data points. 
 
```python
a2 = np.random.uniform(0.4, 0.8, m)
b2 = 3 * a2 + 5 + np.random.normal(0, 0.1, m)

a3 = np.random.uniform(0.2, 0.6, int(m * 0.7))
b3 = 3 * a3 + 7 + np.random.normal(0, 0.1, int(m * 0.7))

a4 = np.random.uniform(0.0, 0.4, m)
b4 = 3 * a4 + 10 + np.random.normal(0, 0.1, m)

a = np.concatenate([a1, a2, a3, a4])
b = np.concatenate([b1, b2, b3, b4])

A = np.column_stack([a, b])
```

Then visualising the data we will see the following:

![k=4 m=100 n=1](/images/2025-07-17-nc-ns-clr-1.svg)

Now to be clear, this is likely not to be a very realistic example. However, this particular example is actually something that works well for the algorithm we will be implementing.

The reason for this is the example we have created is actually a statistical phenomenon called Simpson's Paradox. Which is defined as:

> Simpson’s Paradox is a statistical phenomenon where an association between two variables in a population emerges, disappears or reverses when the population is divided into subpopulations.
 
This is verbatim from [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/paradox-simpson/) so if you are interested in this please do check it out. 

The reason this is a good example is because the true relationship between `$a^i$` and `$b_i$` can not be captured by a single linear regression model. For example, if we were to naively fit a single linear regression model to the data we would get the following:

![k=4 m=100 n=1 fitted line](/images/2025-07-17-nc-ns-clr-2.svg)

This indicates that the relationship is negatively correlated. However, this isn't true and we know this because we can see each cluster has a positive relationship between `$a^i$` and `$b_i$`. 

To solve this problem, we need to fit multiple linear regression models to each cluster individually to capture the true relationship. This is exactly what clusterwise linear regression is designed to do.

### Objective Function

For the clusterwise linear regression problem, we want to minimise the overall fit using:

`$$h(x^j, y_j, a^i, b_i) = |⟨x^j, a^i⟩ + y_j - b_i|^p$$`

where `$h$` is the error function and `$p$` represents the norm used to measure the regression error in the objective function, in this case we care only about `$p=2$` which is the squared error.

We can write this error function in code as follows:

```python
def h(x, y, a, b, p=2):
    return np.abs(np.inner(x, a) + y - b) ** p
```

The function computes the inner product between `$⟨x^j, a^i⟩$` and together `$⟨x^j, a^i⟩ + y_j$` represents the prediction made by the `$j$`-th cluster's regression model for the `$i$`-th data point. The objective function is the total sum of the error function for all data points, which is written as:

`$$f_k(\mathbf{x},\mathbf{y}) = \sum_{i=1}^{m} \min_{j=1,\ldots,k} h(\mathbf{x}^j, y_j, a^i, b_i)$$`

where we minimise `$f_k(\mathbf{x},\mathbf{y})$` subject to: `$\mathbf{x} \in \mathbb{R}^{k \times n}$`  and `$\mathbf{y} \in \mathbb{R}^k$`. 

In code we can write this as:

```python
def obj(x, y, a, b, p=2):
    return np.sum(h(x, y, a, b, p=p))
```

## Incremental Algorithm

The proposed incremental algorithm from Bagirov et al. (2013) is a direct improvement over the original Späth algorithm. It is an improvement because it is able to find significantly better starting points which is a limitation of the original Späth algorithm which relies on random initialisation.

In a previous <a href="/blog/2025-06-27-modified-global-k-means" target="_blank" rel="noopener noreferrer">post</a> we discussed the modified global `$k$`-means algorithm proposed by Bagirov[^3] which is the same incremental approach used in this paper but modified to solve the clusterwise linear regression problem.

Now for those who do not know the modified global `$k$`-means algorithm, that is ok as I will discuss the incremental approach in detail again, for the sake of completeness. 

### Incremental Steps

To understand the algorithm, I found that it is best to just go through each step one by one and see how it works. Then, at the end, we can consolidate the steps into a single function.

#### Solving for `$k=1$`
 
In the case of `$k=1$`, we are simply fitting a single linear regression model to the data. This is a standard linear regression problem and we have already shown what this looks like [earlier](#dataset).

For our regression model, we use the same approach as we did <a href="/blog/2025-07-11-spath-cluster-wise-regression/#the-simple-case" target="_blank" rel="noopener noreferrer">here</a> where we use the following function:

```python
def lingress(a, b, intercept=True):
    if intercept:
        a = np.vstack([a.flatten(), np.ones(len(a))]).T
    return np.linalg.inv(a.T @ a) @ a.T @ b
```

But do note that this regression function is brittle (I am not going into the why in this post), and if you are going to apply this to real data, you should go ahead and use the `scikit-learn` library, specifically the `LinearRegression` class, which you can read about [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

For `$k=1$`:

```python
past = {1: lingress(A[:, 0], A[:, 1])}
```

Since we are solving for `$k=1$`, we have a single cluster and a single regression model. Therefore, the cost is the sum of the squared errors for all data points. 

If we plot this we get the following:

![k=1 m=100 n=1 solve for k=1](/images/2025-07-17-nc-ns-clr-3.svg)

We can also check the cost of this solution:

```python
print(obj(past[1][0], past[1][1], A[:, 0], A[:, 1], p=2))
```

```python
Cost: 196.65
```

#### Solving for `$k>1$`

Solving for the first `$k$` is the simplest case but when we try solve for `$k>1$` we need a better approach to know where to place the new regression line. 

Bagirov et al. (2013) proposed an auxiliary function to help us answer this question, which is very similar in both structure and purpose to the <a href="/blog/2025-06-27-modified-global-k-means/#auxiliary-function" target="_blank" rel="noopener noreferrer">auxiliary function</a> in the modified global `$k$`-means algorithm.

The auxiliary function in the paper is defined as:

`$$\tilde{f}_k(u,v) = \sum_{(a,b) \in A} \min\{r_{ab}^{k-1}, h(u,v,a,b)\}$$`

where `$r_{ab}^{k-1}$` is the regression error for point `$(a,b)$` from our existing `$k-1$` regression lines.







[^1]: Bagirov, A. M., Ugon, J., & Mirzayeva, H. (2013). Nonsmooth nonconvex optimization approach to clusterwise linear regression problems. European Journal of Operational Research, 229(1), 132–142. https://doi.org/10.1016/j.ejor.2013.02.059

[^2]: Späth, H. (1979). Algorithm 39 Clusterwise linear regression. Computing, 22(4), 367–373. https://doi.org/10.1007/bf02265317

[^3]: Adil M. Bagirov, "Modified global k-means algorithm for minimum sum-of-squares clustering problems," *Pattern Recognition* 41, no. 10 (2008): 3192-3199, https://doi.org/10.1016/j.patcog.2008.04.004.

‌

