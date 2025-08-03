---
title: Expecting HTH
subtitle: 
author: 
date: '2025-08-02'
tags:
  - Probability
  - Expected Value
  - Conditional Expectation
---

Problem:

> On average, how many tosses of a fair coin does it take to see HTH?

---

This was honestly me for the past hour..

![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2I5MHZ6M2pieDNtdHZoOHFoY3VuYXMxNW1sc3lrcjFtY2R0OW4zbSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/BBkKEBJkmFbTG/giphy.gif)

To solve this problem we start by defining our initial state `$S_0$` as:

`$$\begin{align*}
S_0 &= P(H)[1 + S_H] + P(T)[1 + S_0] \quad \text{where } P(H) = P(T) = \frac{1}{2}
\end{align*}$$`

then from state `$S_0$` we can define the following states:

`$$\begin{align*}
S_H &= P(H)[1 + S_H] + P(T)[1 + S_{HT}] \\
S_{HT} &= P(H)[1 + 0] + P(T)[1 + S_0]
\end{align*}$$`

The sequence HTH is determined by the state `$S_{HT}$`, if we roll tails we go back to state `$S_0$`. Now we want to convert these equations into standard form to solve for the unknowns.

Starting with `$S_0$` we have:

`$$\begin{align*}
S_0 &= \frac{1}{2}[1 + S_H] + \frac{1}{2}[1 + S_0] \\
S_0 &= \frac{1}{2} + \frac{1}{2} S_H + \frac{1}{2} + \frac{1}{2} S_0 \\
S_0 &= 1 + \frac{1}{2} S_H + \frac{1}{2} S_0 \\
S_0 - \frac{1}{2} S_0 - \frac{1}{2} S_H &= 1 \\
\frac{1}{2} S_0 - \frac{1}{2} S_H &= 1 \\
\frac{1}{2} S_0 - \frac{1}{2} S_H + 0S_{HT} &= 1
\end{align*}$$`

Then doing the same for `$S_H$`:

`$$\begin{align*}
S_H &= \frac{1}{2}[1 + S_H] + \frac{1}{2}[1 + S_{HT}]\\
S_H &= \frac{1}{2} + \frac{1}{2} S_H + \frac{1}{2} + \frac{1}{2} S_{HT} \\
S_H &= 1 + \frac{1}{2} S_H + \frac{1}{2} S_{HT} \\
S_H - \frac{1}{2} S_H - \frac{1}{2} S_{HT} &= 1  \\
\frac{1}{2} S_H - \frac{1}{2} S_{HT} &= 1  \\
0S_0 + \frac{1}{2} S_H - \frac{1}{2} S_{HT} &= 1  \\
\end{align*}$$`

And finally for `$S_{HT}$`:

`$$\begin{align*}
S_{HT} &= \frac{1}{2}[1 + 0] + \frac{1}{2}[1 + S_0] \\
S_{HT} &= \frac{1}{2} + \frac{1}{2} + \frac{1}{2}S_0 \\
S_{HT} &= 1 + \frac{1}{2}S_0 \\
-\frac{1}{2}S_0 + 0S_H + S_{HT} &= 1 \\
\end{align*}$$`

If we set up our system of equations in matrix form we have:

`$$\begin{align*}
\begin{bmatrix}
0.5 & -0.5 & 0 \\ 
0 & 0.5 & -0.5 \\
-0.5 & 0 & 1
\end{bmatrix} 
\begin{bmatrix}
S_0 \\
S_H \\
S_{HT} \\
\end{bmatrix} =
\begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix}
\end{align*}$$`

Ok to solve this we can use Gaussian elimination to solve for the unknowns, we can use NumPy specifically `np.linalg.solve` to do this but since the problem is easy enough we can do it manually.

```python
import numpy as np

A = np.array([[0.5, -0.5, 0], [0, 0.5, -0.5], [-0.5, 0, 1]])
b = np.array([1, 1, 1])

matrix = np.column_stack((A, b))
matrix[0, :] = matrix[0, :] * 2
matrix[1, :] = matrix[1, :] * 2
matrix[-1, :] = matrix[-1, :] * 2
matrix[0, :] = matrix[0, :] + matrix[1, :]
matrix[1, :] = matrix[0, :] + matrix[1, :]
matrix[1, :] = matrix[-1, :] + matrix[1, :]
matrix[0, :] = matrix[0, :] * 2
matrix[0, :] = matrix[0, :] + matrix[-1, :]
matrix[-1, :] = matrix[0, :] + matrix[-1, :]
matrix[-1, :] = matrix[-1, :] * 1 / 2

print(matrix)
```

```python
[[ 1.  0.  0. 10.]
 [ 0.  1.  0.  8.]
 [ 0.  0.  1.  6.]]
```

The final solution is 10 that is from `$S_0$` on average it will take about 10 tosses to see HTH.

`$$\begin{align*}
\begin{bmatrix}
1 & 0 & 0 \\ 
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} 
\begin{bmatrix}
S_0 \\
S_H \\
S_{HT} \\
\end{bmatrix} =
\begin{bmatrix}
10 \\
8 \\
6 \\
\end{bmatrix}
\end{align*}$$`