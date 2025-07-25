---
title: High or Die
subtitle: 
author: 
date: '2025-07-23'
tags:
  - Probability
  - Expected Value
  - Geometric Distribution
  - Law of Total Expectation
---

Problem:

>Francisco rolls a fair die and records the value he rolls. Afterwards, he continues rolling the die until he obtains a value at least as large as the first roll. 
>
>Let`$N$` be the number of rolls after the first roll.
>
> Find `$E[N]$`.

---

Let `$X_i$` be the value of the `$i$`th roll where `$X_i \in \{1, 2, 3, 4, 5, 6\}$` and each have an equal probability of occurring. 

The expected number of rolls required after the first roll depends entirely on `$X_1$`. To model a single roll, we can use the geometric distribution where the expected value is given by:

`$$E[N] = \frac{1}{p}$$`

where `$p$` is the probability of success.

For example, if `$X_1 = 1$`, the next roll is guaranteed to be greater than or equal to `$1$` so the expected number of rolls is `$1$` but how about when `$X_1 = 2$`?

To solve this case, we count the number of sides which are greater than or equal to 2 which is `$\{2, 3, 4, 5, 6\}$` which is of length of 5. This means the probability of rolling a value greater than or equal to 2 is going to be 5 over 6.

`$$E[N] = \frac{1}{(\frac{5}{6})} = 1.2$$`


and for `$X_1 = 3$` we have `$\{3, 4, 5, 6\}$` which is of length 4 giving us a probability of `$\frac{4}{6}$`.

`$$E[N] = \frac{1}{(\frac{4}{6})} = 1.5$$`

Now to solve the entire problem, we can use this exact same process but for each possible outcome of `$X_1$` and then sum the results:

```python
E = 0

for i in range(1, 7):
    E += (1/(7 - i))

print(E)
```

```python
2.45
```

To understand why we can sum the expected values of the random variables is due to the law of total expectation which you can learn more about [here](https://www.youtube.com/watch?v=vJAG4EzSQZA).



