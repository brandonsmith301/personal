---
title: Absolute Difference
subtitle: 
author: 
date: '2025-08-03'
tags:
  - Combinatorics
---

Problem:

> Let `$x_0 = 0$` and let `$x_1, x_2, \dots, x_{10}$` satisfy `$|x_i - x_{i-1}| = 1$` for `$1 ≤ i ≤ 10$` and `$x_{10} = 4$`. 
> 
> How many such sequences are there satisfying these conditions?

My first solution was a brute force, using the following:

```python
import itertools

PATHS = []
for i in itertools.product([-1, 1], repeat=10):
    if sum(i) == 4:
        PATHS.append(i)

print(len(PATHS))
```

```python
120
```

But when doing so, I figured out that the problem was a lot simpler than I thought. We must get to 4, and we can only move 1 step at a time either up or down.

Let `$k$` denote the number of up moves, and `$n$` denote the number of down moves.

We must have `$k - n = 4$` and if we solve for `$k$` and `$n$`, we get `$k = 7$` and `$n = 3$`.

```python
A = np.array([[1,-1], [1,1]], dtype=np.float64)
b = np.array([4, 10], dtype=np.float64)

matrix = np.column_stack((A, b))
matrix[0,:] += matrix[1,:]
matrix[0,:] *= 0.5
matrix[1,:] -= matrix[0,:]
print(matrix)
```

```python
[[ 1.  0.  7.]
 [ 0.  1.  3.]]
```

So we have 7 up moves and 3 down moves (gauss elimination is maybe overkill but I prefer solving systems of equations like this).

The total number of paths is going to be the number of ways to choose 7 up moves out of 10 which can compute by using `math.comb` from the `math` module:

```python
print(math.comb(10, 7))
```

```python
120
```

Of course the brute force solution gives the same answer but it won't scale as well if we had a much larger number of steps.