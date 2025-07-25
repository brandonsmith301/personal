---
title: German Tanks
subtitle: 
author: 
date: '2025-07-25'
tags:
  - Statistics
  - Probability
  - Events
---

Problem:

>Suppose that German tanks are assigned distinct serial numbers `$1, 2, \ldots, N$`. 
>You observe 6 tanks with serial numbers `$38, 17, 59, 42, 97, 120$`. 
>
>Under a frequentist approach, what is the best guess for `$N$`?

---

*If TL;DR, the solution is 139 and turns out there is a much easier way to solve this problem which is by using the minimum-variance unbiased estimator (which I only just read about).*

I like to start by solving a much simpler problem where we know `$N$` and then we can generalise to the case where we don't know `$N$`. 

***Simple Case***

Let `$N = 5$` and we sample two tanks with serial numbers 1 and 2 where the population is:

 `$$\text{all possible tanks} = \{1, 2, 3, 4, 5\}$$`
 
Then we have `$X_2$` representing the maximum serial number of the two tanks observed and then we want to find `$P(X_2 = k)$` for all possible values of `$k$`. 

Since we need two tanks to be sampled, the minium value of `$k$` must be 2 since we can't have a maximum serial number of 1 for two tanks and of course the maximum value of `$k = N$` which in this case is 5.

For `$N = 5$` there are 10 possible outcomes given by the number of ways to choose 2 tanks from 5. Now if we start with `$k = 2$` we have only one possible outcome where the maximum serial number is 2:

`$$P(X_2 = 2) = \frac{1}{10}$$`

Then for `$k = 3$` we have two possible outcomes where the maximum serial number is 3:

`$$P(X_2 = 3) = \frac{2}{10}$$`

Because of this pattern, we can see that:

`$$P(X_2 = k) = \frac{k - 1}{10}$$`

What this pattern is telling us is that when the maximum serial number is `$k$` we have `$k - 1$` possible outcomes which are favourable. 

If we want to generalise `$n$` tanks from 1 to `$N$` when the maximum is `$k$` then we need to make sure to include tank `$k$` and then choose the remaining `$n - 1$` tanks from 1 to `$k - 1$`.

The number of ways to do this is going to be:

`$$\binom{k-1}{n-1}$$`

This means that the generic formula for any `$k$` and `$n$` is:

`$$P(X_n = k) = \frac{\binom{k-1}{n-1}}{\binom{N}{n}}$$`

where the denominator is the number of ways to choose `$n$` tanks from `$N$` (remembering that `$n$` is the sample size).

***Solution***

Now instead of `$N = 5$` we don't know `$N$` and instead of `$n = 2$` we have `$n = 6$` and instead of the serial numbers `$1, 2$` we have `$38, 17, 59, 42, 97, 120$`.

The maximum of our observed sample is 120, so we want to find `$N$` such that the expected value of `$X_6$` equals 120. 

To do this, we can use the generic formula we found earlier:

`$$P(X_6 = k) = \frac{\binom{k-1}{5}}{\binom{N}{6}}$$`

Then find the expected value of `$X_6$` by summing the product of each possible value of `$k$` and its probability:

`$$E[X_6] = \sum_{k=6}^{N} k \cdot P(X_6 = k)$$`

such that the solution is:

`$$E[X_6] = 120$$`

We can solve this numerically by running an experiment from `$n$` to `$1000$` and seeing when the expected value of `$X_6$` is close to 120.

```python
import math

g = lambda N, n, k: math.comb(k - 1, n - 1) / math.comb(N, n)
E = lambda N, n: sum(k * g(N, n, k) for k in range(n, N + 1))

n = 6
k = 120

for N in range(n, 1000):
    e = E(N, n)
    if abs(e - k) < 0.01:
        print("N:       ", N)
        print("Expected:", e)
        break
```

```python
N:        139
Expected: 120.0
```

We found that the best guess for `$N$` is 139.

