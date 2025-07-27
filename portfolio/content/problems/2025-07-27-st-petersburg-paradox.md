---
title: St. Petersburg Paradox
subtitle: 
author: 
date: '2025-07-27'
tags:
  - Probability
  - Expected Value
  - Decision Theory
---

Problem:
>Suppose you are offered to play a game where you flip a fair coin until you obtain a heads for the first time. 
>
>If the first heads occurs on the `$n$`-th flip, you are paid out `$\$2^n$`. 
>
>What is the fair value of this game?

---

As mentioned in the title..

![paradox](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2cxbzg2ODFqODM2YThqZHF4ajUzcGVsa2JqNzF4cHR0M3c3dG0zMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/AxxxG0emVqJPYx57Si/giphy.gif)

I don't actually know the *why* yet so maybe we just try solve the problem first and figure it out together.

***Solution***

First we need to work out the probability of the first heads occurring on the `$n$`-th flip. This means that the first `$n-1$` flips must be tails and the `$n$`-th flip must be heads.

The probability of the first heads occurring on the `$n$`-th flip is:

`$$P(X = n) = \left(0.5\right)^{n-1} \cdot \left(0.5\right)^{1} = \left(0.5\right)^n$$`


Next we want to workout the expected payout of the game which we can write as:

`$$\begin{align*}E[X] &= \sum_{n=1}^{\infty} P(X = n) \cdot 2^n 
\end{align*}$$`

But if we drill into the following `$P(X = n) \cdot 2^n$` we get:

`$$\begin{align*}P(X = n) \cdot 2^n  &= \left(0.5\right)^n \cdot 2^n \\
&= \left(0.5 \cdot 2\right)^n \\
&= 1^n
\end{align*}$$`

Then plugging this back into the expected value we get:

`$$\begin{align*}E[X] &= \sum_{n=1}^{\infty} 1^n \\
&= \infty
\end{align*}$$`

So the expected payout of the game is infinite! 

***Why is this a paradox?***

This famous paradox was introduced by Nicolaus Bernoulli in 1713 and the key question is *"How much should one be willing to pay for playing this game?"*

The expected payout doesn't rely on how much you are willing to pay for the game but dependent on when we get the first heads. If the payout is infinite then we should be willing to pay any amount of money to play the game. 

But this isn't quite right.

The best way to think about this is to model the problem computationally and see what happens. The program which we used to model the problem is below:

```python
rng = np.random.default_rng()

p_ave = []
h_ave = []

for k in range(1, 1000, 10):
    player = []
    host = []
    for _ in range(10000):
        n = 0
        while True:
            x = rng.choice([0, 1])
            n += 1
            if x == 1:
                break
        player.append(2**n - k)
        host.append(k - 2**n)
    p_ave.append(np.mean(player))
    h_ave.append(np.mean(host))
```

Then if we plot the average profit of the player and the host we get the following graph:

![st-petersburg-paradox](/images/st-petersburg-paradox.svg)

This simulation shows that while the expected payout of the game is infinite, the average profit of the player is only positive for small values of `$k$` and then becomes negative for larger values of `$k$`.

This is a mathematically perfect game with infinite expected value that cannot function realistically because both the host and players would rationally refuse to agree on what is *fair*.

