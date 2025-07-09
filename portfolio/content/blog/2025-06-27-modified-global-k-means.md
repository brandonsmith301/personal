---
title: Modified Global k-means Algorithm for Minimum Sum-of-Squares Clustering Problems
subtitle: 
author: 
date: '2025-06-27'
tags:
  - Clustering
  - Nonsmooth optimisation
---

---

This post focuses on understanding and re-implementing the paper Modified Global `$k$`-means Algorithm for Minimum Sum-of-Squares Clustering Problems by Adil M. Bagirov[^1]. 

The objective of clustering is to group data points based on their relationships into distinct clusters, which is a form of unsupervised learning because we don't explicitly tell the algorithm what the clusters should be. 

The paper looks at the global `$k$`-means, which is a variant of the standard `$k$`-means. Both algorithms solve the same clustering objective but use different optimisation strategies. I will leave out the *why* for now as it will make a lot more sense as you read on. 

***Note:*** I've implemented most of the code using explicit loops rather than NumPy vectorisation (as much as possible), making the algorithmic steps more visible. In practice, vectorising these operations would be more efficient.


## Problem Definition

As introduced in the paper, the clustering problem assumes we are given a finite set `$A$` of `$m$` data points in `$\mathbb{R}^n$`:
`$$A = \{a^1, \dots, a^m\} \text{ where } a^i \in \mathbb{R}^n, \, i = 1, \dots, m$$`

This can also be represented as an `$m \times n$` matrix where each row corresponds to a data point. In Python we can write this as:

```python
# A is a 100 × 2 matrix (100 points in 2D space)
m = 100
n = 2   
A = np.random.rand(m, n)
```

While the matrix representation is computationally convenient, it's important to point out that sets are unordered collections, whereas matrices impose a specific row ordering. I put this here to prevent us from confusing these as equivalent data structures when they are fundamentally different.

The authors consider the hard unconstrained partition clustering problem. This involves partitioning the set `$A$` into `$k$` disjoint subsets with respect to the following criteria:

- `$A^j \neq \emptyset \text{ for } j = 1, \ldots, k$` (each cluster must be non-empty)

- `$A^j \cap A^l = \emptyset \text { for } j \neq l$` (clusters are disjoint - no point belongs to multiple clusters)

- `$A = \bigcup_{j=1}^{k} A^j$` (every point in `$A$` belongs to exactly one cluster)

Each cluster `$A^j$` can be identified by its center `$x^j \in \mathbb{R}^n$` for `$j = 1, \ldots, k$`.

### Dataset

Given that we have defined the problem we are working with, we can now generate a concrete example where `$m = 100$` and `$n = 2$`.

![Example of Clustering Problem](/images/2025-06-27-modified-global-k-means-1.svg)

We will be using this problem throughout this post since it is easy to understand and, with `$n = 2$`, it is easy to visualise. Typically, we don't have it this easy. 

The code below will generate the same problem we are working with:

```python
# Generate dataset with 3 clusters
k = 3
m = 100
n = 2

A, clusters = sklearn.datasets.make_blobs(
    n_samples=m,
    centers=k,
    n_features=n,
    random_state=1
)
```

### Objective Function

The function below guides the optimisation for the clustering problem:

`$$\psi_k(x, w) = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} w_{ij} \|x^j - a^i\|^2$$`

Where the goal is to find the best partition of `$m$` data points into `$k$` clusters by minimising the total distance between points and their assigned cluster centers. In the objective function `$\psi_k$` we are introduced to the following component `$w_{ij} \|x^j - a^i\|^2$`.

The first part of the component `$w_{ij}$` is the binary assignment matrix, that is:

`$$w_{ij} = \begin{cases}
1 & \text{if pattern } a^i \text{ is allocated to cluster } j \\
0 & \text{otherwise}
\end{cases}$$`

This means if we were to randomly initalise `$w$`:

```python
w = np.zeros((m, k))
for i in range(m):
    j = np.random.randint(0, k)
    w[i, j] = 1

# shows first 4 rows
w[:4]
```

```python
[[0. 1. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
```

The term `$\|x^j - a^i\|^2$` is the squared Euclidean distance between point `$a^i$` and center `$x^j$`. 

For example, if we have the following point and center:

```python
x = np.array([[0.1], [0.2]])
a = np.array([[0.3], [0.4]])

dist = np.sum((x - a) ** 2)
```

Then the squared Euclidean distance is:

![Example of Euclidean distance](/images/2025-06-27-modified-global-k-means-2.svg)

Finally, we can write the objective function as:

``` python
def obj(x, w, A, m, k):
    """
    x: Cluster centers x^j
    w: Assignment matrix w_ij
    A: a^i in A
    m: Number of data points
    k: Number of clusters
    """
    cost = 0
    for i in range(m):
        for j in range(k):
            if w[i, j] == 1:
                cost += w[i, j] * np.sum((x[j] - A[i]) ** 2)
    return cost / m
```

We also put together the functions which will be used to initialise the assignment matrix `$w$` and cluster centers `$x$`. Both of these are initialised randomly.

```python
def init_w(m, k):
    w = np.zeros((m, k))
    for i in range(m):
        j = np.random.randint(0, k)
        w[i, j] = 1
    return w

def init_x(A, m, k):
    return A[np.random.randint(0, m, size=k)]
```

If we use our randomly initialised assignment matrix `$w$` and cluster centers `$x$` we get the following cost:

```python
w = init_w(m, k)
x = init_x(A, m, k)
costs = obj(x, w, A, m, k)
print(f"Cost: {costs:.4f}")
```

```python
Cost: 110.6423
```

This value is what we want to minimise through the optimisation process.

### Constraints

The constraints here are verbatim from Bagirov's paper. 

***Cluster Centers***

The cluster centers `$x$` are represented as a matrix `$x \in \mathbb{R}^{n \times k}$`, where each column `$x^j \in \mathbb{R}^n$` (`$j = 1, \dots, k$`) corresponds to a center in `$n$`-dimensional space.

***Assignment Matrix***

The assignment weights `$w$` form a matrix `$w \in \{0, 1\}^{m \times k}$`, where each row `$w_{i:}$` satisfies:

`$$\sum_{j=1}^k w_{ij} = 1, \quad \forall i \in \{1, \dots, m\}$$`

This means that each data point is assigned to exactly one cluster. 

***Binary Assignment***

Each entry `$w_{ij}$` is binary:

`$$w_{ij} \in \{0, 1\}, \quad \forall i, j$$`

Which is the same as the definition of the assignment matrix `$w$` we introduced earlier.

`$$w_{ij} = \begin{cases}
1 & \text{if pattern } a^i \text{ is allocated to cluster } j \\
0 & \text{otherwise}
\end{cases}$$`

## k-means with Random Initialisation

From most of the definitions we have introduced so far, we can now write the `$k$`-means algorithm with random initialisation.

We just need to include two steps:

1. The update step for the cluster centers `$x$` which works by:

- Taking the current assignments `$w$` matrix
- Computing new cluster centers as the mean of assigned points
- Updating `$x^j$` positions

```python
def update_x(w, A):
    """
    w: Assignment matrix w_ij
    A: a^i in A
    """
    d = w.sum(axis=0).reshape(-1, 1)
    if np.any(d == 0):
        res = np.zeros_like(w.T @ A)
        mask = d != 0
        m_flat = mask.flatten()
        res[m_flat] = (w.T @ A)[m_flat] / d[mask]
        return res
    else:
        return w.T @ A / d
```

If we take the initial cluster centers `$x$` and update them we get the following cost:

```python
updated_x = update_x(w, A)
costs = obj(updated_x, w, A, m, k)
print(f"Cost: {costs:.4f}")
```

```python
Cost: 40.6800
```
These new updated cluster centers bring the cost down after updating the cluster centers. We can visualise the before and after cluster centers:

![Before and after cluster centers](/images/2025-06-27-modified-global-k-means-3.svg)

2. The update step for the assignment matrix `$w$` which works by:

`$$w_{ij} = \begin{cases}
1 & \text{if } j = \arg\min_{l \in \{1,\ldots,k\}} \|x^l - a^i\|^2 \\
0 & \text{otherwise}
\end{cases}$$`

This is the standard nearest-neighbour assignment rule which assigns point `$a^i$` to the cluster center `$x^j$` that minimises `$\|x^j - a^i\|^2$`.

```python
def update_w(a, x, m, k):
    """
    a: a^i in A
    x: Cluster centers x^j
    m: Number of data points
    k: Number of clusters
    """
    w = np.zeros((m, k))
    for i in range(m):
        distances = [np.sum((x[j] - a[i]) ** 2) for j in range(k)]
        nearest_cluster = np.argmin(distances)
        w[i, nearest_cluster] = 1
    return w
```

This is why when first updating the cluster centers, the centers look like they are in the wrong place. We actually need to first update the assignment matrix `$w$` and then the cluster centers `$x$`.

```python
updated_w = update_w(A, x, m, k)
updated_x = update_x(updated_w, A)
costs = obj(updated_x, updated_w, A, m, k)
print(f"Cost: {costs:.4f}")
```

```python
Cost: 19.7920
```

And if we visualise the before and after cluster centers we get the following:

![Before and after cluster centers (after updating the assignment matrix `$w$` and then the cluster centers `$x$`)](/images/2025-06-27-modified-global-k-means-4.svg)

Now both the cluster centers and the assignment matrix `$w$` have been updated. We can see that the cluster centers look a bit better but not *perfect*. This is because we are running only one iteration of the algorithm.

The full algorithm is as follows:

```python
def kmeans_iteration(A, x, m, k, max_iters=100):
    """
    A: a^i in A
    x: Cluster centers x^j
    m: Number of data points
    k: Number of clusters
    max_iters: Maximum number of iterations
    """
    x_copy = x.copy()
    for _ in range(max_iters):
        updated_w = update_w(A, x_copy, m, k)
        updated_x = update_x(updated_w, A)
        x_copy = updated_x

    return x_copy, updated_w
```

If we run the algorithm for 100 iterations we get the following cost and output:

```python
updated_x, updated_w = kmeans_iteration(A, x, m, k, max_iters=100)
costs = obj(updated_x, updated_w, A, m, k)
print(f"Cost: {costs:.4f}")
```

```python
Cost: 6.3645
```

While the cost has decreased, the cluster centers look like they are still in the wrong place. This is because `$k$`-means is a local search algorithm and it tends to get stuck in local minima. 

![Before and after cluster centers (after 100 iterations)](/images/2025-06-27-modified-global-k-means-5.svg)

Even if we increase the number of iterations, the cluster centers will not improve and will stay in this local minimum. To improve the algorithm, we could try different *random* initialisations and then run the algorithm on the best initialisation.

This means the improved algorithm would look like:

```python
MULTISTART = 10
best_cost = np.inf
best_x = None
best_w = None

for _ in range(MULTISTART):
    w = init_w(m, k)
    x = init_x(A, m, k)
    costs = obj(x, w, A, m, k)
    if costs < best_cost:
        best_cost = costs
        best_x = x
        best_w = w

updated_x, updated_w = kmeans_iteration(
    A, best_x, m, k, max_iters=100
)
costs = obj(updated_x, updated_w, A, m, k)
print(f"Cost: {costs:.4f}")
```

```python
Cost: 1.5628
```

This is a much better result than the previous one.

![Before and after cluster centers (after 100 iterations and MULTISTART=10)](/images/2025-06-27-modified-global-k-means-6.svg)

The improvement is due to the *multistart* heuristic and is a common technique for optimisation problems. However, a much better improvement can be made by using `$k$`-means++[^2], which you can read about [here](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf).

## Incremental Approach

The incremental approach avoids poor local minima by building up the clusters incrementally. Instead of starting with `$k$` clusters, we start with `$k-1$` clusters and solve for the `$k$`-th cluster at a time.  

What this means is that if we have a good (not necessarily optimal) solution for `$k-1$` clusters, we can find a better solution for `$k$` clusters by incrementally optimising the `$k$`-th center position.

We will now look at two algorithms that use this incremental approach, which will then lead us to the modified global `$k$`-means algorithm proposed by Bagirov.

### Global `$k$`-means

The algorithm is a significant improvement over `$k$`-means as it exhaustively tries every single point as a potential starting position for the new `$k$`-th center. This algorithm was proposed by Likas et al.[^3] in 2003.

The code for the global `$k$`-means algorithm is as follows:

```python
def global_kmeans(A, m, k):
    """
    A: a^i in A
    m: Number of data points
    k: Number of clusters
    max_iters: Maximum number of iterations
    """
    past_x = {1: np.mean(A, axis=0).reshape(1, -1)}

    for i in range(2, k + 1):
        best_cost = np.inf
        best_x = None
        best_w = None

        for a in A:
            x = np.vstack((past_x[i-1], a.reshape(1, -1)))
            x, _ = kmeans_iteration(A, x, m, i, max_iters=100)

            w = update_w(A, x, m, i)
            cost = obj(x, w, A, m, i)
            if cost < best_cost:
                best_cost = cost
                best_x = x
                best_w = w

        past_x[i] = best_x

    return best_x, best_w
```

To better understand the algorithm, we will solve the problem for `$k=3$` step by step.

The first step is to solve for `$k=1$`, where the optimal solution is just going to be the mean of the points in `$A$`.

```python
past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
```

This will give us the first cluster center `$x^1$`:

![First cluster center `$x^1$`](/images/2025-06-27-modified-global-k-means-7.svg)

For `$k=2$`, we exhaustively try every point as the starting position for the second center, here we set `$i=2$` and then loop through each point in `$A$`:

```python
best_cost = np.inf
best_x = None

i = 2

for a in A:
    x = np.vstack((past_x[i - 1], a.reshape(1, -1)))
    x, w = kmeans_iteration(A, x, m, i, max_iters=100)

    cost = obj(x, w, A, m, i)
    if cost < best_cost:
        best_cost = cost
        best_x = x

past_x[2] = best_x
```

which gives us the cluster center `$x^2$`:

![Second cluster center `$x^2$`](/images/2025-06-27-modified-global-k-means-8.svg)

and if we run the same code but with `$i=3$`:

```python
best_cost = np.inf
best_x = None

i = 3

for a in A:
    x = np.vstack((past_x[i - 1], a.reshape(1, -1)))
    x, w = kmeans_iteration(A, x, m, i, max_iters=100)

    cost = obj(x, w, A, m, i)
    if cost < best_cost:
        best_cost = cost
        best_x = x

past_x[3] = best_x
```

we will get our final cluster center `$x^3$`:

![Third cluster center `$x^3$`](/images/2025-06-27-modified-global-k-means-9.svg)

### Fast Global `$k$`-means

The fast global `$k$`-means algorithm (proposed in the same paper as the global `$k$`-means algorithm) addresses a significant limitation of the original algorithm, which is having to run `$k$`-means `$m$` times (once for each potential center), making it *very* impractical for larger problems.

The key improvement is that instead of actually computing the objective function for every possible `$k$`-th center, we can estimate which center would give the maximum reduction in the objective function using this formula:

`$$b_j = \sum_{i=1}^{m} \max \{0, d^i_{k-1} - \|a^j - a^i\|^2\}$$`

In this formula, `$d^i_{k-1}$` represents the distance from point `$a^i$` to its nearest center among the existing `$k-1$` clusters which is given by:

`$$d^i_{k-1} = \min\{\|x^1 - a^i\|^2, \ldots, \|x^{k-1} - a^i\|^2\}$$`

This is the first component of the formula and it outputs an array of length `$m$`:

```python
def component_1(A, current_x, m):
    """
    A: a^i in A
    current_x: x^1, ..., x^{k-1}
    m: Number of data points
    """
    dist = np.zeros(m)
    for i in range(m):
        min_dist = np.inf
        for center in current_x:
            min_dist = min(min_dist, np.sum((center - A[i]) ** 2))
        dist[i] = min_dist
    return dist
```

The second component of the formula computes `$\|a^j - a^i\|^2$`, which represents the potential new assignment cost for point `$a^i$` if assigned to the new center at location `$a^j$` and like component 1 this outputs an array of length `$m$`:

```python
def component_2(A, d, m):
    """
    A: a^i in A
    d: d^i_{k-1}
    m: Number of data points
    """
    b = np.zeros(m)
    for j in range(m):
        b_j = 0
        for i in range(m):
            improve = d[i] - np.sum((A[j] - A[i]) ** 2)
            b_j += max(0, improve)
        b[j] = b_j
    return b
```

To show how this works, we will go through step by step starting with `$k=1$`. We start by initialising the first cluster center `$x^1$` as the mean of the points in `$A$`:

```python
past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
```

Then we compute the distance for each point in `$A$` to the first cluster center `$x^1$`:

```python
d = component_1(A, past_x[1], m)
```

By plotting the distance for each point in `$A$` we can visualise how far each point is from the first cluster center `$x^1$` where the darker the shade the farther the point is from the center:

![Distance from first cluster center `$x^1$`](/images/2025-06-27-modified-global-k-means-10.svg)

Now that we have `$d^i_{k-1}$`, we can compute `$b_j$` to give us the sum of all positive improvements:

```python
b = component_2(A, d, m)
```
Similarly, we can visualise the second component of the formula:

![Second component of the formula](/images/2025-06-27-modified-global-k-means-11.svg)

This time the darker shade tell us the us the estimated improvement for each point if we were to add a new center at that point. 

To then find the potential new center, we need to find the point with the maximum improvement, which we can do by taking the maximum value of `$b_j$`:

```python
past_x[2] = np.vstack((past_x[1], A[np.argmax(b)]))
```

and we now have the second cluster center `$x^2$`:

![Second cluster center `$x^2$`](/images/2025-06-27-modified-global-k-means-12.svg)

The next step is to run `$k$`-means on the new cluster centers for `$k=2$`:

```python
i = 2
best_x, best_w = kmeans_iteration(A, past_x[i], m, i, max_iters=100)
```

Here we set `$i=2$` because we are now solving for `$k=2$` and now we can visualise the optimised cluster centers:

![Optimised cluster centers for `$k=2$`](/images/2025-06-27-modified-global-k-means-13.svg)

We then repeat the process for `$k=3$` by computing the minimum distance for each point in `$A$` to our `$k-1$` cluster centers (`$x^1$` and `$x^2$`) and then finding the point with the maximum improvement:

```python
# assign our optimised centers to past_x[2]
past_x[2] = best_x
d = component_1(A, past_x[2], m)
b = component_2(A, d, m)
```

and if we plot the output:

![Finding k=3 cluster center](/images/2025-06-27-modified-global-k-means-14.svg)

By looking at the plot we can quite easily see what neighbourhood the potential new center will be in. If we now pick the point with the maximum improvement we get the estimated cluster center `$x^3$`:

```python
past_x[3] = np.vstack((past_x[2], A[np.argmax(b)]))
```

![Third cluster center `$x^3$`](/images/2025-06-27-modified-global-k-means-15.svg)

and then run `$k$`-means on the new cluster centers for `$k=3$`:

```python
i = 3
best_x, best_w = kmeans_iteration(A, past_x[i], m, i, max_iters=100)
```

which gives us the optimised cluster centers for `$k=3$`:

![Optimised cluster centers for `$k=3$`](/images/2025-06-27-modified-global-k-means-16.svg)

We can now put this all together to form the fast global `$k$`-means algorithm:

```python
def fast_global_kmeans(A, m, k):
    """
    A: a^i in A
    m: Number of data points
    k: Number of clusters
    """
    past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
    for i in range(2, k + 1):
        d = component_1(A, past_x[i - 1], m)
        b = component_2(A, d, m)
        x = np.vstack((past_x[i - 1], A[np.argmax(b)]))
        x, w = kmeans_iteration(A, x, m, i, max_iters=100)
        past_x[i] = x
    return past_x[k], w
```

Now to see how much faster the fast global `$k$`-means algorithm is than the global `$k$`-means algorithm, we can run the two algorithms on the same problem and compare the time taken:

```python
%%timeit
best_x, best_w = global_kmeans(A, m, k)

14.7 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
%%timeit
best_x, best_w = fast_global_kmeans(A, m, k)

179 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

The difference in time is quite significant, with the fast global `$k$`-means algorithm being about 80 times faster than the global `$k$`-means algorithm to achieve the same result.

### Exact Improvement

This was fascinating for me to read and very satisfying once I understood it. The result from the formula:

`$$b_j = \sum_{i=1}^{m} \max \{0, d^i_{k-1} - \|a^j - a^i\|^2\}$$`

is not an estimate but in fact the exact improvement in the objective function. To understand why this is the case, we will remind ourselves of the formula for the objective function:

`$$\psi_k(x, w) = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} w_{ij} \|x^j - a^i\|^2$$`

Now we will reformulate the objective function in terms of non-smooth non-convex optimisation. This means that instead of using the assignment matrix `$w_{ij}$` explicitly, we are using implicit assignments through the min operator.

`$$f_k(x) = \frac{1}{m} \sum_{i=1}^{m} \min_{j=1,...,k} ||x^j - a^i||^2$$`

Bagirov shows that `$b_j = f_{k-1} - f_k$`, which means that `$b_j$` is the exact improvement in the objective function. I will now explain why this is the case step by step.

***STEP 0:***

We start by reminding ourselves that we are working with the `$(k-1)$`-clustering problem with centers `$x^1, x^2, \dots, x^{k-1}$`.

For each point `$a^i$` (where `$a \in A$` and `$A$` represents the problem dataset), we know that the current assignment costs (which are the costs of assigning `$a^i$` to a potential cluster) are given by `$\min\{\|x^1-a^i\|^2, \dots, \|x^{k-1}-a^i\|^2\}$`, which is represented as `$d^i_{k-1}$`. 

For every data point `$a_j$` in the dataset `$A$`, we evaluate it as a potential location for the new `$k$`-th center.

There will be a lot of fancy notation, but I have tried to decompose it into code to make it more intuitive. We will start with the original problem we have been working with:

```python
past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
d = component_1(A, past_x[1], m)
b = component_2(A, d, m)

past_x[2] = np.vstack((past_x[1], A[np.argmax(b)]))
```
Remember this is just the first and second step of the fast global `$k$`-means algorithm without the `$k$`-means optimisation step.

***STEP 1:***

We define index sets to differentiate between points that will benefit from the new center versus those that will remain with their current assignments.

`$$\begin{align*}
I_1 &= \{i \in \{1, \dots, m\} : \|a^i - a^j\|^2 \geq d^i_{k-1}\}, \\
I_2 &= \{i \in \{1, \dots, m\} : \|a^i - a^j\|^2 < d^i_{k-1}\}
\end{align*}$$`

Here `$I_1$` represents points that *don't* switch, which are the points where the new center `$a^j$` is farther away than their current nearest center.

The partition between these two sets is complete and exclusive, meaning that `$I_1 \cup I_2$` will give every point up to size `$m$` and `$I_1 \cap I_2$` will give the empty set.

By visualising an example, we can see that the points in `$I_1$` are the points that would not benefit from the new center `$a^j$`, and the points in `$I_2$` are the points that would benefit from the new center `$a^j$`, which are coloured in grey.

![Index sets `$I_1$` and `$I_2$`](/images/2025-06-27-modified-global-k-means-17.svg)

The points in `$I_2$` represent those that will switch their assignment to the new center `$a^j$`, since `$a^j$` is closer to them than their current nearest center `$x^1$`.

The code below will give us the index sets `$I_1$` and `$I_2$` for the new center `$a^j$`:

```python
func = lambda i, aj: np.sum((A[i] - aj) ** 2)

I1 = [i for i in range(m) if func(i, past_x[2][-1]) >= d[i]]
I2 = [i for i in range(m) if func(i, past_x[2][-1]) < d[i]]
```

***STEP 2:***

For a `$(k-1)$`-clustering, the objective function is:

`$$f_{k-1}(x) = \frac{1}{m} \sum_{i=1}^{m} \min_{j=1,\ldots,k-1} \|x^j - a^i\|^2$$`

and since `$d^i_{k-1} = \min_{j=1,\ldots,k-1} \|x^j - a^i\|^2$`, we can rewrite the objective function as:

`$$f_{k-1}(x) = \frac{1}{m} \sum_{i=1}^{m} d^i_{k-1}$$`

which in code is:

```python
fk_1 = lambda m, d: 1 / m * np.sum(d)
```

We then reorganise the sum using `$I_1$` and `$I_2$` to update the objective function: 

`$$f_{k-1} = \frac{1}{m} \left( \sum_{i \in I_1} d^i_{k-1} +\sum_{i \in I_2} d^i_{k-1} \right)$$`

which in code is:

```python
fk_2 = lambda m, d, I1, I2: 1 / m * (np.sum(d[I1]) + np.sum(d[I2]))
```

Both functions which I have defined above are equivalent to one another. If we compare the two outputs, we can see that they are the same:

```python
np.equal(fk_1(m, d), fk_2(m, d, I1, I2))
```

which gives us:

```python
True
```

By decomposing the objective function, it makes it easier to see the differences between the current state and future state of our optimisation problem. 

***STEP 3:***

When we add the new center `$a^j$`, each point chooses its nearest center from all `$k$` centers:

`$$f_{k} = \frac{1}{m} \sum_{i=1}^{m} \min\{d^i_{k-1}, \|a^j - a^i\|^2\}$$`

By definition, the min operator behaves differently for our index sets `$I_1$` and `$I_2$`.

Therefore, we can rewrite `$f_k$` as:
`$$f_{k} = \frac{1}{m} \left( \sum_{i \in I_1} d^i_{k-1} + \sum_{i \in I_2} \|a^j - a^i\|^2 \right)$$`

This is because for points in `$I_1$`, since `$\|a^j - a^i\|^2 \geq d^i_{k-1}$`, the min is `$d^i_{k-1}$`, and for points in `$I_2$`, since `$\|a^j - a^i\|^2 < d^i_{k-1}$`, the min is `$\|a^j - a^i\|^2$`. 

While obvious, I have added the explanation for clarity.

***STEP 4:***

Now we will show `$f_{k-1} - f_k$` which is the exact improvement in the objective function:

`$$f_{k-1} - f_{k} = \frac{1}{m} \left( \sum_{i \in I_1} d^i_{k-1} +\sum_{i \in I_2} d^i_{k-1} \right) - \frac{1}{m} \left( \sum_{i \in I_1} d^i_{k-1} +\sum_{i \in I_2} \|a^j - a^i\|^2 \right)$$`

and since `$\sum_{i \in I_1} d^i_{k-1}$` appears in both expressions, they cancel out. This leaves us with:

`$$f_{k-1} - f_{k} = \frac{1}{m} \left( \sum_{i \in I_2} d^i_{k-1} - \sum_{i \in I_2} \|a^j - a^i\|^2 \right)$$`

then after factoring out the common sum, we get:

`$$f_{k-1} - f_{k} = \frac{1}{m} \sum_{i \in I_2} \left(d^i_{k-1} - \|a^j - a^i\|^2 \right)$$`

***STEP 5:***

If we look back at the upper bound estimate:

`$$b_j = \sum_{i=1}^{m} \max\{0, d^i_{k-1} - \|a^j - a^i\|^2\}$$`

for the points in `$I_1$`, we know that it was constructed based on the condition:

`$$\|a^j - a^i\|^2 \geq d^i_{k-1}$$`

which means that `$d^i_{k-1} - \|a^j - a^i\|^2$` must be less than or equal to `$0$`, and for the points in `$I_2$`, the condition was:

`$$\|a^j - a^i\|^2 < d^i_{k-1}$$`

which means `$d^i_{k-1} - \|a^j - a^i\|^2$` is greater than `$0$`.

This means that points in `$I_1$` contribute nothing to `$b_j$`, while points in `$I_2$` contribute their full improvement to `$b_j$`. 

Therefore, we can write:
`$$b_j = \sum_{i \in I_2} (d^i_{k-1} - \|a^j - a^i\|^2)$$`

and from Step 4 we showed that:
`$$f_{k-1} - f_k = \frac{1}{m} \sum_{i \in I_2} (d^i_{k-1} - \|a^j - a^i\|^2)$$`

which implies:
`$$b_j = m(f_{k-1} - f_k)$$`



## Modified GKM

The modified global `$k$`-means algorithm builds on the fast global `$k$`-means algorithm by adding continuous optimisation and is the final algorithm we will cover, which is the proposed algorithm in the paper.

### Auxiliary Function

Bagirov introduces an auxiliary function which is used to find better starting points. This function is defined as:

`$$\bar{f}_{k}(y) = \frac{1}{m} \sum_{i=1}^{m} \min\{d^i_{k-1}, \|y - a^i\|^2\}$$`

Unlike the previous approach that restricts potential centers to discrete data points `$a^j$`, the function `$\bar{f}_{k}(y)$` uses a continuous variable `$y$` that can be positioned anywhere in the feature space. 

This allows us to find optimal cluster centers that may lie between data points, which should provide better initialisation for the `$k$`-means algorithm since we are not restricting to the discrete dataset locations.

In code, the auxiliary function is defined as:

```python
def auxiliary_function(y, A, d):
    costs = 0.0
    for i, a_i in enumerate(A):
        costs += min(d[i], np.sum((y - a_i) ** 2))
    return costs / len(A)
```

### Partitioning the Space

Since `$y$` can be positioned anywhere in continuous space rather than being restricted to discrete points in `$A$`, we need to understand how different locations affect the clustering. 

To better understand this, we partition the entire space into two regions based on the impact of placing a center at location `$y$`. 

This gives us two sets:

`$$\bar{D} = \left\{\, y \in \mathbb{R}^n : \| y - a^i \|^2 \leq d^i_{k-1} \;\; \forall i \in \{1, \ldots, m\} \,\right\}$$`

Where `$\bar{D}$` represents the region where placing any new center would benefit every single data point. While in theory this is possible, this region is typically empty or very small in practice.  

The second set `$D_0$` is the complement of `$\bar{D}$`.

`$$
\begin{align*}
D_0 &= \mathbb{R}^n \setminus \bar{D} \\
    &\equiv \left\{\, y \in \mathbb{R}^n : \exists\, I \subset \{1, \ldots, m\},\, I \neq \emptyset : \|y - a^i\|^2 < d^i_{k-1} \;\; \forall i \in I \,\right\}
\end{align*}
$$`

Which means that there exists at least one subset of data points such that `$y$` is closer to all points in that subset than their current centers are. While the construction of `$D_0$` looks complicated, we have already seen the same computation in the fast global `$k$`-means algorithm with `$b_j$`. 

The only difference is that instead of taking the maximum improvement we are interested in finding all improvements greater than `$0$`. 

```python
past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
d = component_1(A, past_x[1], m)
b = component_2(A, d, m)

d0 = [j for j in range(m) if b[j] > 0]
```

Now that we know how to construct `$D_0$`, we can look at the three sets that we will be using to partition the space.

For any `$y \in D_0$`, we consider the following sets:

`$$
\begin{align*} \displaystyle
S_1(y) &= \{ a_i \in A : \| y - a_i \|_2 = d_{i}^{k-1} \} \\
S_2(y) &= \{ a_i \in A : \| y - a_i \|_2 < d_{i}^{k-1} \} \\
S_3(y) &= \{ a_i \in A : \| y - a_i \|_2 > d_{i}^{k-1} \}
\end{align*}
$$`

***Set 1***

The first set `$S_1$` are the boundary points which are indifferent from their current nearest cluster center and `$y$`. Which means whether they switch or not, it doesn't matter as the cost remains the same.

`$$\begin{align*} 
S_1(y) &= \{ a_i \in A : \| y - a_i \|_2 = d_{i}^{k-1} \}
\end{align*}$$`

This is another set which isn't a likely case in practice but is still important to consider since it's theoretically possible.

***Set 2***

The second set `$S_2$` are the points that would switch since the points here are closer to `$y$` than their current nearest center.

`$$
\begin{align*} \displaystyle
S_2(y) &= \{ a_i \in A : \| y - a_i \|_2 < d_{i}^{k-1} \}
\end{align*}
$$`

If you remember our definition of `$I_2$` from the fast global `$k$`-means algorithm, this is the same set except we are now using the continuous variable `$y$` instead of the discrete points `$a^i$`.

***Set 3***

Then finally, we have `$S_3$` which is the set which contains points which stay since they are closer to their current nearest center then `$y$`.

`$$
\begin{align*} \displaystyle
S_3(y) &= \{ a_i \in A : \| y - a_i \|_2 > d_{i}^{k-1} \}
\end{align*}
$$`

Which we have also already seen in the fast global `$k$`-means algorithm with `$I_1$`.

### The Full Algorithm

The modified global `$k$`-means algorithm's improvement is to find the location `$y^* \in D_0$` that minimises the auxiliary function:

`$$\bar{f}_{k}(y^*) = \min_{y \in D_0} \bar{f}_{k}(y)$$`

where `$y^*$` is the best starting point for the new center.

The key insight is that for any `$y \in D_0$`, the set `$S_2(y)$` is non-empty and contains exactly the points that would benefit from placing a center at `$y$`.

We can now start to write the code for the modified global `$k$`-means algorithm.

To start off we start by solving for `$x^1$` which is the mean of the data points. This follows the same pattern as both the global and fast global `$k$`-means algorithms.

```python
past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
d = component_1(A, past_x[1], m)
b = component_2(A, d, m)
```

Then we find `$d_0$`, which contains the points that would benefit from the new center. These are simply the points where `$b > 0$` (using the exact formula for `$b_j$`).

```python
d0 = [j for j in range(m) if b[j] > 0]
```

We then iterate over each point in `$d_0$` and find the set `$S_2(y)$`. We then take the mean of the points in `$S_2(y)$` and evaluate using the auxiliary function. We then take the point with the lowest cost as our starting point.

```python
best_cost = np.inf
starting_point = None

for j in d0:
    s2 = [i for i in range(m) if np.sum((A[j] - A[i]) ** 2) < d[i]]
    y = np.mean(A[s2], axis=0)
    cost = auxiliary_function(y, A, d)

    if cost < best_cost:
        best_cost = cost
        starting_point = y
```

Once we have found the best starting point using the auxiliary function, we can iterate on the starting point to find `$y^*$` which is the optimal center.

```python
MAX_ITER = 10

y = starting_point.copy()

for _ in range(MAX_ITER):
    _s2 = [i for i in range(m) if np.sum((y - A[i]) ** 2) < d[i]]

    if _s2 == s2:
        break

    if len(_s2) > 0:
        y = np.mean(A[_s2], axis=0)

    s2 = _s2
```

The algorithm stops when no more points join or leave `$S_2$`, indicating that we have found the optimal continuous location `$y^*$` for the `$k$`-th center. 

We can then use this optimal location as the starting point for the `$k$`-means algorithm.

```python
best_x, best_w = kmeans_iteration(A, y, m, i, max_iters=100)
```

The next step follows the same pattern as the previous algorithms we have covered, which is the incremental approach. However, we will stop with `k=2` and consolidate the code we have written so far.

We can start by defining a function to find the set `$S_2$`:

```python
def get_s2(A, y, d, m):
    return [i for i in range(m) if np.sum((A[i] - y) ** 2) < d[i]]
```

Then we can define a function to find the initial starting point `$y$` using the auxiliary function:

```python
def initial_y(A, d, d0):
    best_cost = np.inf
    best_y = None
    m = len(A)

    for j in d0:
        s2 = get_s2(A, A[j], d, m)

        if s2:
            y = np.mean(A[s2], axis=0)
        else:
            y = A[j]

        cost = auxiliary_function(y, A, d)
        if cost < best_cost:
            best_cost = cost
            best_y = y

    return best_y
```

And then finally we can define a function to improve the starting point so that we can find the optimal center `$y^*$`:

```python
def improve_y(A, starting_point, d, max_iter=100):
    y = starting_point.copy()
    m = len(A)
    prev_s2 = []

    for _ in range(max_iter):
        s2 = get_s2(A, y, d, m)

        if set(s2) == set(prev_s2):
            break

        if s2:
            y = np.mean(A[s2], axis=0)

        prev_s2 = s2

    return y
```

The modified global `$k$`-means algorithm follows a similar incremental approach to the fast global `$k$`-means algorithm, there are two key differences: how we find the best starting point and how we exit the algorithm.

We have seen how to find the best starting point in the previous section, as for exiting the algorithm, we check when the relative improvement is less than the tolerance.

Bagirov suggests a tolerance between `$0.01$` and `$0.1$` and noted that large tolerance values cause the algorithm to stop early, resulting in fewer but larger clusters, whereas small values cause it to continue too long, producing artificial clusters.

The complete algorithm is as follows:

```python
def modified_global_kmeans(A, max_k=20, tol=0.05, max_iter=100):
    """
    A: Dataset points a^i in A
    max_k: Maximum number of clusters to try
    tol: Tolerance for relative improvement
    max_iter: Max no of iterations for k-means
    """
    past_x = {1: np.mean(A, axis=0).reshape(1, -1)}
    m = len(A)
    w = init_w(m, 1)

    f1 = obj(x=past_x[1], w=w, A=A, m=m, k=1)
    f_prev = f1
    k = 1

    for i in range(2, max_k + 1):
        d = component_1(A, past_x[i - 1], m)
        b = component_2(A, d, m)
        d0 = [j for j in range(m) if b[j] > 0]
        
        if len(d0) == 0:
            break

        tmp_y = initial_y(A, d, d0)
        y = improve_y(A, tmp_y, d, max_iter=max_iter)
        x = np.vstack((past_x[i - 1], y.reshape(1, -1)))
        x, w = kmeans_iteration(A, x, m, i, max_iters=max_iter)

        fk = obj(x=x, w=w, A=A, m=m, k=i)
        relative_improvement = (f_prev - fk) / f1

        if relative_improvement < tol:
            break

        past_x[i] = x
        f_prev = fk 
        k = i

    return past_x[k], w
```

## Comparison

We will first run the initial comparison we did between the fast global `$k$`-means algorithm and the global `$k$`-means algorithm but this time we will include the modified global `$k$`-means algorithm. 

The problem dataset is the same as the one we have been using throughout this blog post which is a 3 cluster dataset with 100 points in 2 dimensions.

```python
%%timeit
best_x, best_w = global_kmeans(A, m, k)

14.8 s ± 570 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
%%timeit
best_x, best_w = fast_global_kmeans(A, m, k)

226 ms ± 35.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
%%timeit
best_x, best_w = modified_global_kmeans(A, tol=0.01)

323 ms ± 112 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

We can see that the modified global `$k$`-means algorithm is slightly slower than the fast global `$k$`-means algorithm but still significantly faster than the global `$k$`-means algorithm. All three algorithms find the same solution for this simpler problem.

This time we will test the algorithms on a more complex dataset: 6 clusters, 500 points, 15 dimensions, with standard deviation 3.0 and centers in (-5, 5), creating cluster overlap.

```python
k = 6
m = 500
n = 15

A, clusters = make_blobs(
    n_samples=m,
    centers=k,
    n_features=n,
    cluster_std=3.0,  
    center_box=(-5, 5),  
    random_state=1
)

# ----
best_x, best_w = global_kmeans(A, m, k)
print(obj(x=best_x, w=best_w, A=A, m=m, k=len(best_x)))

best_x, best_w = fast_global_kmeans(A, m, k)
print(obj(x=best_x, w=best_w, A=A, m=m, k=len(best_x)))

best_x, best_w = modified_global_kmeans(A, tol=0.01)
print(obj(x=best_x, w=best_w, A=A, m=m, k=len(best_x)))
```

```python
133.059
133.061
128.059
```

In this case we find that the modified global `$k$`-means algorithm had the best performance in terms of improving the objective function. This is what we would expect since the algorithm is designed to be better at finding the optimal centers. 

Bagirov's paper actually performed a much more thorough comparison of the algorithms through multiple datasets - so please do take a look if you are interested in the details. 

## Closing Thoughts

The reconstruction of Bagirov's paper was a great way to understand not only the proposed modified global `$k$`-means algorithm but also both the global and fast global `$k$`-means algorithms. 

While the `$k$`-means algorithm is taught at the undergraduate level, I found it was looked at only briefly. This paper allowed me to understand the algorithm in a more detailed way and also the different ways we can improve it.

All the code for this post will be available at some point in the future. I hope you found this post interesting, and if you have any questions or feedback, please feel free to reach out through [LinkedIn](https://www.linkedin.com/in/brandonsmith--/).

[^1]: Adil M. Bagirov, "Modified global k-means algorithm for minimum sum-of-squares clustering problems," *Pattern Recognition* 41, no. 10 (2008): 3192-3199, https://doi.org/10.1016/j.patcog.2008.04.004.

[^2]: David Arthur and Sergei Vassilvitskii, "k-means++: The advantages of careful seeding", https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

[^3]: Likas, A., Vlassis, N., & Verbeek, J. J. (2003). The global k-means clustering algorithm. Pattern Recognition, 36(2), 451-461. https://doi.org/10.1016/S0031-3203(02)00060-2.