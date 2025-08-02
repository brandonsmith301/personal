---
title: Nonsmooth Nonconvex Optimization Approach to Clusterwise Linear Regression Problems
subtitle: 
author:
date: '2025-07-17'
draft: false
tags:
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

a1 = np.random.uniform(0.0, 1.0, m)
b1 = 3 * a1 + 2 + np.random.normal(0, 0.35, m)

A = np.column_stack([a1, b1])
```

However, this only generates a single cluster of data points. In our example we will solve `$k=4$` which means we will need to generate 3 more clusters of data points. 
 
```python
a2 = np.random.uniform(0.0, 1.0, m)
b2 = 3 * a2 + 5 + np.random.normal(0, 0.35, m)

a3 = np.random.uniform(0.0, 1.0, m)
b3 = 3 * a3 + 7 + np.random.normal(0, 0.35, m)

a4 = np.random.uniform(0.0, 1.0, m)
b4 = 3 * a4 + 10 + np.random.normal(0, 0.35, m)

a = np.concatenate([a1, a2, a3, a4])
b = np.concatenate([b1, b2, b3, b4])
A = np.column_stack([a, b])
```

Then visualising the data we will see the following:

![k=4 m=100 n=1](/images/2025-07-17-nc-ns-clr-1.svg)

What we can see from this is that if we were to naively fit a single linear regression model to the data we would get the following:

![k=4 m=100 n=1 fitted line](/images/2025-07-17-nc-ns-clr-2.svg)

To solve this problem, we need to fit multiple linear regression models to each cluster individually to capture each cluster's relationship. 

This is exactly what clusterwise linear regression is designed to do.

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

### First Steps

***Solving for `$k=1$`***
 
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

***Solving for `$k>1$`***

Solving for the first `$k$` is the simplest case but when we try solve for `$k>1$` we need a better approach to know where to place the new regression line. 

### The Auxiliary Function

Bagirov et al. (2013) proposed an auxiliary function to help us answer this question, which is very similar in both structure and purpose to the <a href="/blog/2025-06-27-modified-global-k-means/#auxiliary-function" target="_blank" rel="noopener noreferrer">auxiliary function</a> in the modified global `$k$`-means algorithm.

The auxiliary function in the paper is defined as:

`$$\tilde{f}_k(u,v) = \sum_{(a,b) \in A} \min\{r_{ab}^{k-1}, h(u,v,a,b)\}$$`

where `$r_{ab}^{k-1}$` is the regression error for point `$(a,b)$` from our existing `$k-1$` regression lines. In fact, the regression error is akin to `$d_i^{k-1}$` in the modified global `$k$`-means algorithm which is the distance from point `$i$` to its closest existing center.

To show this auxiliary function, we can write the following code:

```python
def auxiliary_function(u, v, A, errors):
    costs = 0.0
    for i in range(len(A)):
        e = h(u, v, A[i, 0], A[i, 1])
        costs += min(errors[i], e)
    return costs
```

The function takes in the new regression line `$(u,v)$` and the matrix `$A$` which contains `$(a^i, b_i)$`. It also needs the errors computed from our `$k-1$` regression lines. 

To compute this, we can write the following code:

```python
errors = []
for i in range(len(A)):
    e = h(past[1][0], past[1][1], A[i, 0], A[i, 1])
    errors.append(e)
```

The errors we are computing is the squared error between the predicted value and the actual value. If we plot the errors as the colour of our data points, we get the following:

![k=1 m=100 n=1 solve for k=1](/images/2025-07-17-nc-ns-clr-4.svg)

The darker points show where the data points are poorly fit by our current regression line and are good candidates for a second regression line.

### Partitioning the Space

With the defined auxiliary function, we want to: 

`$$\min_{u,v} \tilde{f}_k(u,v) \text{ subject to } u \in \mathbb{R}^n, v \in \mathbb{R}$$`

This optimisation problem is nonconvex and nonsmooth with potentially many local minima. Specifically, Bagirov et al. (2013) noted that:

>This problem is nonconvex and therefore it may have a large number of local solutions.

Now similar to the discussion regarding <a href="/blog/2025-06-27-modified-global-k-means/#partitioning-the-space" target="_blank" rel="noopener noreferrer">partitioning the space</a> in the modified global `$k$`-means algorithm we also want to partition the space for the clusterwise linear regression problem into two regions denoted by sets `$C_k$` and `$\overline{C_k}$`.

Our first set `$C_k$` contains all hyperplanes that are worse than or equal to the previous iteration for every single data point. 

`$$C_k = \{(u,v) \in \mathbb{R}^{n+1} : h(u,v,a,b) > r_{ab}^{k-1} \quad \forall (a,b) \in A\}$$`

This means that any starting point in this set is useless since the objective function is constant over this set. Fortunately, we have another set `$\overline{C_k}$` which is constructed as follows:

`$$\overline{C_k} = \{(u,v) \in \mathbb{R}^{n+1} : \exists(a,b) \in A : h(u,v,a,b) < r_{ab}^{k-1}\}$$`

Any hyperplane in `$\overline{C_k}$` is better than our previous iteration for at least one data point. This means we want to start strictly in this set as it guarantees that our objective function will decrease.

### Finding Initial Solutions

From above we know that we want to start in `$\overline{C_k}$` but since this set is defined over the continuous space `$\mathbb{R}^{n+1}$` we can't use it directly as there are infinitely many points in this set. Instead, we want to use the points in `$A$` to guide the search through the space.

If you recall, we are in iteration `$k$` of the algorithm, so we already have `$k-1$` regression lines from previous iterations. Each data point `$(a,b)$` is currently assigned to whichever of these `$k-1$` lines fits it best.

To compute the best fit, we can write the following code:

```python
def get_best_fit(a, b, past):
    best_j = None
    error = np.inf

    for j, (x, y) in past.items():
        e = h(x, y, a, b)
        if e < error:
            error = e
            best_j = j

    return best_j
```

Then we want to create a hyperplane parellel to the best fit such that it passes through the point `$(a,b)$`. This is constructed as follows:

`$$x_{ab} = x_j \quad \text{and} \quad y_{ab} = b - \langle x_j, a \rangle$$`

This means we are using the same slope as the best fit but we are using the point `$(a,b)$` to compute the intercept. Computationally, we can write this as:

```python
def get_init_solutions(A, past, errors):
    candidates = []
    for a, b in A:
        j = get_best_fit(a, b, past)
        x_ab = past[j][0].copy()
        y_ab = b - x_ab * a

        B_ab = []
        for k, (c, d) in enumerate(A):
            if h(x_ab, y_ab, c, d) < errors[k]:
                B_ab.append([c, d])

        if len(B_ab) >= 2:
            B_ab = np.array(B_ab)
            x_ab, y_ab = lingress(B_ab[:, 0], B_ab[:, 1])

        candidates.append((x_ab, y_ab))

    return candidates
```

To see this in action, we can start by using the previous function to generate the canidate for only the first data point `$(a_1,b_1)$`:

```python
for a, b in A[:1, :]:
    j = get_best_fit(a, b, past)
    x_ab = past[j][0].copy()
    y_ab = b - x_ab * a
```

Then we can plot the candidate line as well as highlighting the point `$(a_1,b_1)$`:

![k=1 m=100 n=1 solve for k=1](/images/2025-07-17-nc-ns-clr-5.svg)

If it is not already clear, the list of `errors` we pass into `get_init_solutions` is `$r_{ab}^{k-1}$` for each data point `$(a,b)$` in `$A$`. 

There is another part to the function which constructs the set `$B_{ab}$` which the authors of the paper denote as:

`$$B_{ab} = \{(c,d) \in A : h(x^{ab}, y_{ab}, c, d) < r_{cd}^{k-1}\}$$`

and is defined as:

> The set `$B_{ab}$` contains all points from the set `$A$` attracted by the clusterwise linear regression function `$(x^{ab}, y_{ab})$`.

Now instead of just using the first data point `$(a_1,b_1)$`, we can use the entire dataset `$A$`:

```python
candidates = get_init_solutions(A, past, errors)
```

This now contains the candidate lines for each data point in `$A$` but only if the candidate line is better than the previous `$k-1$` lines.

### Refining the Candidate Lines

With these new candidate lines, we can compute the improvement, given by:

`$$d(x^{ab}, y_{ab}) = \tilde{f}_{k-1} - \tilde{f}_k = \sum_{(c,d) \in A} \max\{0, r_{cd}^{k-1} - h(x^{ab}, y_{ab}, c, d)\}$$`

We write this in code as:

```python
def get_d(u, v, A, errors):
    d = 0
    for i, (a, b) in enumerate(A):
        e = h(u, v, a, b)
        d += max(0, errors[i] - e)
    return d
```

These improvements tell us how much would the total error decrease if we were to use this candidate line instead of the previous `$k-1$` lines. 

If we were to run this function for each candidate line:

```python
decreases = []
for candidate in candidates:
    d = get_d(candidate[0], candidate[1], A, errors)
    decreases.append(d)
```

We would be able to plot the improvements as the colour of our data points:

![k=1 m=100 n=1 solve for k=1](/images/2025-07-17-nc-ns-clr-7.svg)

With this new list of decreases, we can start to filter out the candidate lines that are not good enough. Rather than absolute thresholds, the authors propose a relative threshold. 

This is defined as:

> Let `$\gamma_1 \in [0, 1]$` be a given number. 
>
> Define `$\overline{d}_1 = \max\{d(x^{ab}, y_{ab}) : (a,b) \in A\}$` 
> 
> and the following set:
> 
> `$\overline{A}_1 = \{(a,b) \in A : d(x^{ab}, y_{ab}) \geq \gamma_1 \overline{d}_1\}$`.

What this means is using the list `decreases` (which we computed above) we want to get the max value then use this to create a threshold which is done by multiplying with `$\gamma_1$`. 

For example, if we set `$\gamma_1 = 0.9$`:

```python
d1 = max(decreases)
g1 = 0.9

A1 = []
for i, decrease in enumerate(decreases):
    if decrease >= g1 * d1:
        A1.append(i)
```

This filters out about half of the data points and gives us the set `$\overline{A}_1$` which contains only the data points whose parallel candidate lines provide sufficient improvement above our chosen relative threshold.

Now we can plot the improvements for each data point index alongside the threshold.

![k=1 m=100 n=1 solve for k=1](/images/2025-07-17-nc-ns-clr-8.svg)

We keep everything above the threshold which is the set `$\overline{A}_1$` and everything below the threshold is filtered out.

Now for each `$(a,b) \in \overline{A}_1$` we want to find which points the candidate would actually attract. This is done by:

`$$B_{ab} = \{(c,d) \in A : h(x^{ab}, y_{ab}, c, d) < r_{cd}^{k-1}\}$$`

The set `$B_{ab}$` contains all points from the dataset that would get better predictions from our candidate line `$(\overline{x}^{ab}, \overline{y}_{ab})$` than from their current best assignments.

In code, we need to check every data point against each promising candidate:

```python
A2 = []
for i in A1:
    x_ab, y_ab = candidates[i]

    B_ab = []
    for j, (c, d) in enumerate(A):
        if h(x_ab, y_ab, c, d) < errors[j]:
            B_ab.append([c, d])

    if len(B_ab) >= 2:
        B_ab = np.array(B_ab)
        x_ab, y_ab = lingress(B_ab[:, 0], B_ab[:, 1])
        A2.append((x_ab, y_ab))
    else:
        A2.append((x_ab, y_ab))
```

This refinement process creates the set `$\overline{A}_2$` of refined hyperplanes:

`$$\overline{A}_2 = \{(u,v) : u \in \mathbb{R}^n, v \in \mathbb{R} \text{ and } \exists (a,b) \in \overline{A}_1 \text{ s.t. } u = \bar{x}_{ab}, v = \bar{y}_{ab}\}$$`

Then finally, we use the auxiliary function for each refined hyperplane to find the best fit, such that:

`$$\hat{f}_k = \min\{\tilde{f}_k(u,v) : (u,v) \in \overline{A}_2\}$$`

To compute this, we can write the following code:

```python
best = []
for u, v in A2:
    best.append(auxiliary_function(u, v, A, errors))
```

Then using the min value we want to apply a relative threshold, similar to what we have already done. In particular, we need to apply a relative threshold to the set `$\overline{A}_2$` to reduce the number of points. 

The authors proposed the following:

> Let `$\gamma_2 \in [1, \infty)$` be a given number. 

So now using `$\gamma_2$` we can filter out the points that are not good enough. This filtering process creates the set `$\overline{A}_3$` of refined hyperplanes:

`$$\overline{A}_3 = \{(u,v) \in \overline{A}_2 : \hat{f}(u,v) \leq \gamma_2 \hat{f}_{k,min}\}$$`

This entire process from `$A_1$` to `$A_3$` is done to construct the initial solution to solve the auxiliary problem. Below I define the complete algorithm using the code we have written so far which constructs the sets `$A_1$`, `$A_2$`, and `$A_3$`.

```python
def get_A1(A, candidates, errors, g1=0.9):
    decreases = []
    for candidate in candidates:
        d = get_d(candidate[0], candidate[1], A, errors)
        decreases.append(d)

    d1 = max(decreases)

    A1 = []
    for i, decrease in enumerate(decreases):
        if decrease >= g1 * d1:
            A1.append(i)

    return A1


def get_A2(A, A1, candidates, errors):
    A2 = []
    for i in A1:
        x_ab, y_ab = candidates[i]

        B_ab = []
        for j, (c, d) in enumerate(A):
            if h(x_ab, y_ab, c, d) < errors[j]:
                B_ab.append([c, d])

        if len(B_ab) >= 2:
            B_ab = np.array(B_ab)
            x_ab, y_ab = lingress(B_ab[:, 0], B_ab[:, 1])
            A2.append((x_ab, y_ab))
        else:
            A2.append((x_ab, y_ab))

    return A2


def get_A3(A, A2, errors, g2=1):
    best = []
    for u, v in A2:
        best.append(auxiliary_function(u, v, A, errors))

    A3 = []
    for i, (u, v) in enumerate(A2):
        if best[i] <= g2 * min(best):
            A3.append((u, v))

    return A3
```

Then we can put this all together into a single function:

```python
def algorithm3(A, past, errors, g1=0.9, g2=1):
    candidates = get_init_solutions(A, past, errors)
    A1 = get_A1(A, candidates, errors, g1)
    A2 = get_A2(A, A1, candidates, errors)
    A3 = get_A3(A, A2, errors, g2)
    return A3
```

### The Auxiliary Problem

At this point, we have `$\overline{A}_3$` containing the best candidate hyperplanes. But we still need to solve the auxiliary problem (which the authors call problem 6).

Similar to the discussion regarding how we are going to partition the space, we now want to understand the structure of the auxiliary function around any given hyperplane `$(u,v)$`. The authors introduce two sets:

`$$\mathcal{B}(u,v) = \{(a,b) \in A : h(u,v,a,b) < r_{ab}^{k-1}\}$$`

`$$\overline{\mathcal{B}}(u,v) = \{(a,b) \in A : h(u,v,a,b) = r_{ab}^{k-1}\}$$`

Obviously, the set `$\mathcal{B}(u,v)$` contain the data points that would prefer hyperplane `$(u,v)$` over their current best assignments. While the set `$\overline{\mathcal{B}}(u,v)$` contains data points which are indifferent between the two hyperplanes.

To actually solve the problem, we start with a point `$(u_0,v_0)$` and then iteratively refine it by fitting regression to the set `$\mathcal{B}(u,v)$`, which we only stop when the attracted set has stopped changing.

The full algorithm is as follows:

```python
def algorithm4(A, errors, u0, v0, max_iter=100):
    ul, vl = u0, v0

    for _ in range(max_iter):
        B = []
        for i, (a, b) in enumerate(A):
            if h(ul, vl, a, b) < errors[i]:
                B.append([a, b])

        if len(B) < 2:
            break

        B = np.array(B)

        _u, _v = lingress(B[:, 0], B[:, 1])

        B_new = []
        for i, (a, b) in enumerate(A):
            if h(_u, _v, a, b) < errors[i]:
                B_new.append([a, b])

        if np.array_equal(np.array(B), np.array(B_new)):
            return _u, _v

        ul, vl = _u, _v

    return ul, vl
```

The authors state a proposition which is as follows:

> Algorithm 4 terminates in a finite number of iterations and if the set `$\overline{\mathcal{B}}(u,v) = \emptyset$` then the point `$(u,v)$` is a local minimizer of the function `$\tilde{f}_k$`.

The claim is that algorithm 4 will always stop in a finite number of steps and the authors prove this in the paper.

***But why?***

The algorithm is a descent method which means the auxiliary function strictly decreases at each iteration, unless we have converged at some minimum. 

If we have converged, then:

`$$\mathcal{B}(u_l,v_l) = \mathcal{B}(u_{l+1},v_{l+1})$$`

which is exactly what our `np.array_equal(B, B_new)` check detects in `algorithm4`.

Our dataset has `$m$` points and there are only `$2^m$` possible subsets that could be attracted to any hyperplane. Since there are only finitely many possible attracted sets, and the algorithm descends with each iteration, the algorithm must converge.

The authors also prove that under certain conditions specifically when `$\mathcal{B}(u,v) = \emptyset$`, the converged point is guaranteed to be a local minimum of the auxiliary function. This is the more complex part of the proof that I will cover at some point in the future.

***Formulating the initial solution***

Now if we take a look at what our initial solution would be, given we have already solved `$k=1$`, we can show what `$\mathcal{B}(u,v)$` looks like alongside our initial solution given by `algorithm4`:

![algorithm4](/images/2025-07-17-nc-ns-clr-9.svg)

What the figure shows is the initial solution being fitted to the points in the set `$\mathcal{B}(u,v)$` (the blue points). The remaining points (not coloured blue) belong to `$\overline{\mathcal{B}}(u,v)$` and continue using their existing assignments. 

And of course, this refined solution is then used as *the initial solution* for the `$k$`-th step in the clusterwise linear regression problem.


### The Complete Algorithm

Recall the Späth algorithm which we discussed in <a href="/blog/2025-07-11-spath-cluster-wise-regression/#complete-algorithm" target="_blank">a previous post</a> (code implementation is also there). The limitation of this algorithm is due to poor random initialisation which can lead to suboptimal solutions, which then requires using heuristics such as multiple restarts, also <a href="/blog/2025-06-27-modified-global-k-means/#k-means-with-random-initialisation" target="_blank">discussed here</a>.

The authors in this paper propose a modified version of the Späth algorithm, which is detailed in the paper as Algorithm 5. The work we have done so far constructs only the initial solution, the next step is to integrate this with the Späth algorithm.

***Initialisation***

To initialise the algorithm, we need to find the best fit for the entire dataset:

```python
past = {0: lingress(A[:, 0], A[:, 1])}
```

This specifically solves `$k=1$` for the clusterwise linear regression problem.

***Constructing the initial solution***

The output of `algorithm4` creates the set `$\overline{A}_4$` which can be defined as:

```python
def get_A4(A, A3, errors, max_iter=100):
    A4 = []
    for u0, v0 in A3:
        u, v = algorithm4(A, errors, u0, v0, max_iter=max_iter)
        A4.append((u, v))

    return A4
```

Then we want to construct another set `$\overline{A}_5$` which further filters the points from `$\overline{A}_4$` based on the relative threshold `$\gamma_3$`.

As we recall, the set `$\overline{A}_4$` contains local minima of the auxiliary function. The purpose of `$\overline{A}_5$` is to select the best local minima among all the local minima. 

The set is constructed as:

`$$\overline{A}_5 = \{(\bar{u}, \bar{v}) \in \overline{A}_4 : \bar{f}k(\bar{u}, \bar{v}) \leq \gamma_3 \bar{f}{k,min}\}$$`

and in code we can write it as:

```python
def get_A5(A, A4, errors, g3=10):     
    best = []
    for u, v in A4:
        best.append(auxiliary_function(u, v, A, errors))

    A5 = []
    for i, (u, v) in enumerate(A4):
        if best[i] <= g3 * min(best):
            A5.append((u, v))

    return A5
```

The purpose is that since `algorithm4` is a local optimisation method, it may not always find the best local minima and may find some which aren't worth wasting time on.

So for each `$k$` we want to incrementally add one more regression line by:

```python
for _ in range(2, k + 1):
    errors = []
    for ai, bi in zip(a, b):
        error = np.inf
        for x, y in past.values():
            e = h(x, y, ai, bi)
            if e < error:
                error = e
        errors.append(error)

    A3 = algorithm3(A, past, errors, g1=g1, g2=g2)
    A4 = get_A4(A, A3, errors, max_iter=max_iter)
    A5 = get_A5(A, A4, errors, g3=g3)
```

The hyperparameters `$\gamma_1$`, `$\gamma_2$`, and `$\gamma_3$` control how much strict the fitlering should be. They are defined as `g1`, `g2`, and `g3` in the code. Breaking down the hyperparameters:

| Hyperparameter | Effect |
|:--------------|:-------|
| `$\gamma_1$` | Lower = more candidates = slower but more thorough. |
| `$\gamma_2$` | Higher = more permissive. |
| `$\gamma_3$` | Higher = keeps more local optima. |

For specifics please refer to the paper, section 5.1.

***Refinement of all linear regression functions***

Then for each `$(\bar{u}, \bar{v}) \in \overline{A}_5$` we want to refine the linear regression function. This is done by applying the Späth algorithm which will then give us the set `$\overline{A}_6$` which contain the final candidates for the `$k$`-clusterwise regression problem. 

The complete refinement process is as follows:

```python
def refine(a, b, A5, past):
    A6 = []
    for _, (u, v) in enumerate(A5):
        past_tmp = past.copy()
        past_tmp[len(past)] = (u, v) 
        k = len(past_tmp)
        
        C = np.zeros(len(a), dtype=int)

        for i, (ai, bi) in enumerate(zip(a, b)):
            best = 0
            error = np.inf

            for j, (x, y) in enumerate(past_tmp.values()):
                e = h(x, y, ai, bi)
                if e < error:
                    error = e
                    best = j
            C[i] = best
            
        C = iterative_improvement(a, b, C, k)

        refined = []
        for c_id in range(k):
            mask = C == c_id
            if np.sum(mask) >= 2:
                x, y = lingress(a[mask], b[mask])
                refined.append((x, y))
            else:
                refined.append(list(past_tmp.values())[c_id])

        obj = 0
        for i, (ai, bi) in enumerate(zip(a, b)):
            cluster = C[i]
            x, y = refined[cluster]
            obj += h(x, y, ai, bi)

        A6.append((refined, obj))

    return A6
```

We apply the Späth algorithm to each proposed solution in `$\overline{A}_5$` combined with `$k-1$` previous solutions. This function is called `iterative_improvement` which is the exact same code used in my previous post on the <a href="/blog/2025-07-11-spath-cluster-wise-regression/#complete-algorithm" target="_blank" rel="noopener noreferrer">Späth algorithm</a>.

***Computation of the solution***

The final step is to compute the solution. This is done by iterating through the set `$\overline{A}_6$` and finding the solution with the lowest objective value.

```python
best = 0
best_obj = np.inf
for i, (_, obj) in enumerate(A6):
    if obj < best_obj:
        best_obj = obj
        best = i

for j, (x, y) in enumerate(A6[best][0]):
    past[j] = (x, y)
```

This is the final solution to the clusterwise linear regression problem.

***Algorithm 5***

Putting together the different components we have discussed so far, we can write the complete algorithm as follows:

```python
def algorithm5(a, b, k=3, g1=0.9, g2=1, g3=10, max_iter=100):
    past = {0: lingress(a, b)}
    A = np.column_stack((a, b))
    
    for _ in range(2, k + 1):
        errors = []
        for ai, bi in zip(a, b):
            error = np.inf
            for x, y in past.values():
                e = h(x, y, ai, bi)
                if e < error:
                    error = e
            errors.append(error)

        A3 = algorithm3(A, past, errors, g1=g1, g2=g2)
        A4 = get_A4(A, A3, errors, max_iter=max_iter)
        A5 = get_A5(A, A4, errors, g3=g3)
        A6 = refine(A[:, 0], A[:, 1], A5, past)

        best = 0
        best_obj = np.inf
        for i, (_, obj) in enumerate(A6):
            if obj < best_obj:
                best_obj = obj
                best = i

        for j, (x, y) in enumerate(A6[best][0]):
            past[j] = (x, y)

    return past
```

If we solve the clusterwise linear regression problem with `algorithm5` we can see the following results:

![algorithm5](/images/2025-07-17-nc-ns-clr-10.svg)

This example is of course a very simple one, which the traditional Späth algorithm would have solved just as well. But the authors show that the algorithm is able to solve more complex problems, which is demonstrated in the paper.

## Closing Thoughts

I found that implementing this paper was a lot more challenging than both the <a href="/blog/2025-07-11-spath-cluster-wise-regression/#closing-thoughts" target="_blank" rel="noopener noreferrer">Späth algorithm</a> and the <a href="/blog/2025-06-27-modified-global-k-means/#closing-thoughts" target="_blank" rel="noopener noreferrer">modified global k-means algorithm</a>. The most challenging part was understanding the fitlering process, specifically `$A_1$`, `$A_2$`, and `$A_3$`.

The initial toy example I wanted to use relates to the Simpson's paradox, which is a classic paradox in statistics. The example is as follows:

![simpsons-paradox](/images/2025-07-17-nc-ns-clr-11.svg)

The idea is that when fitting a single regression line to the entire dataset, the line is misleading and if we try to make decisions based on this line, we may be making incorrect decisions.

For example, imagine if we try solve for `$k=1$` for the entire dataset, we would get the following fit:

![simpsons-paradox-1](/images/2025-07-17-nc-ns-clr-12.svg)

This fit indicates that the relationship is negative when in reality it is positive for each group. Now, the idea was to use clusterwise linear regression to solve this problem and fit a different line to each group.

But what I found was that the algorithm was not able to solve this problem, which is due to the fact that when solving incrementally from `$k=1$`, the algorithm becomes constrained by the previous solutions. 

If I solve for this problem using `algorithm5` with `k=4` I get the following results:

![simpsons-paradox-2](/images/2025-07-17-nc-ns-clr-13.svg)

The key problem here is the problem dataset is structurally sparse from the `$k=1$` perspective which creates a wrong baseline for each incremental step of the algorithm. The authors do point out this limitation in the paper:

> However, this algorithm fails when data sets are sparse or have outliers...

This was interesting for me to observe and would be something that could be further explored to work for data structures like the example above. 


[^1]: Bagirov, A. M., Ugon, J., & Mirzayeva, H. (2013). Nonsmooth nonconvex optimization approach to clusterwise linear regression problems. European Journal of Operational Research, 229(1), 132–142. https://doi.org/10.1016/j.ejor.2013.02.059

[^2]: Späth, H. (1979). Algorithm 39 Clusterwise linear regression. Computing, 22(4), 367–373. https://doi.org/10.1007/bf02265317

[^3]: Adil M. Bagirov, "Modified global k-means algorithm for minimum sum-of-squares clustering problems," *Pattern Recognition* 41, no. 10 (2008): 3192-3199, https://doi.org/10.1016/j.patcog.2008.04.004.

‌

