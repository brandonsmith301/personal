---
title: Resell Painting
subtitle: 
author: 
date: '2025-07-21'
tags:
  - Probability
  - Expected Value
---

Problem:

>You are currently bidding for a painting. You know that the value of the painting is between $0 and $100,000 uniformly. 
>
>If your bid is greater than the value of the painting, you win and sell it to an art museum at a price of 1.5 times the value. 
>
>What's your bid to maximize your profit? If you can not profit, bid 0.

---

Brain must not be functioning properly tonight but this problem was actually very simple.

We have `$X \sim \mathcal{U}(0, 100000)$` and our profit is `$1.5X - K$` where `$K$` is our bid. We win the painting if `$X < K$`. 

This means the probability of winning is given by:

`$$P(X < K) = \frac{K}{100000}$$`

We now know that `$X$` follows a uniform distribution between `$0$` and `$K$` as winning means the value of the painting must be at most `$K$`. The expected value requires the mean of the distribution which is given by `$\frac{K}{2}$`.

The expected profit is given by:

`$$E[1.5X - K] = 1.5 \cdot \frac{K}{2} - K$$`

This already tells us that we should bid nothing as the expected profit is always going to be negative. However, negative by how much?

We can find out by simplifying the right hand side:

`$$\begin{align*}
E[1.5X - K] &= 1.5 \cdot \frac{K}{2} - K \\
&= \frac{1.5K}{2} - K \\
&= \frac{1.5K}{2} - \frac{2K}{2} \\
&= \frac{1.5K - 2K}{2} \\
&= \frac{-0.5K}{2} \\
&= -\frac{K}{4} \\
&= -0.25K
\end{align*}$$`

This means for every dollar we bid, we will lose `$0.25$` on average.