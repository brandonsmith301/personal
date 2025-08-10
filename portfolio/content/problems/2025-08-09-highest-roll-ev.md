---
title: Highest Roll Expected Value
subtitle: 
author: 
date: '2025-08-09'
tags:
  - Probability
  - Expected Value
  - Combinatorics
---

Problem:

> Jim will roll a fair, six-sided die until he gets a 4. What is the expected value of the highest number he rolls through this process?

Easiest approach is to simulate the process and see what the average is.

```python
trials = 10_000

max_rolls = []
for _ in range(trials):
    rolls = []
    while True:
        roll = np.random.randint(1, 7)
        rolls.append(roll)
        if roll == 4:
            break
    max_rolls.append(max(rolls))
    
print(np.mean(max_rolls))
```

which should give us a number `$\approx 5.16$`.

Then if you did not know how to solve the problem analytically, you can now use the simulation above to try figure out how we can generalise the approach.

We generate a sequence of rolls until we get a 4 then compute the maximum roll. What this tells us is that any roll below 4 isn't important and we can ignore them entirely.

For `$P(X = 4)$` we only need to consider the case where the first roll is a 4 since the sequence ends at this point. We either roll a 4, 5, or 6 which means the probability of rolling a 4 is `$\frac{1}{3}$`.

As for `$P(X = 5)$` among all possible orderings of when we first see each of {4, 5, 6}, only 1 out of 6 arrangements favours 5 being the maximum:

`$$\begin{align*}
P(X = 5) = 1/6
\end{align*}$$`

Then finally for `$P(X = 6)$` we can either roll a 6 or a 4:

`$$\begin{align*}
P(X = 6) = 1/2
\end{align*}$$`

Then we can compute the expected value of the highest roll:

`$$\begin{align*}
E(X) = 4 \cdot \frac{1}{3} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{2} \approx 5.16
\end{align*}$$`




