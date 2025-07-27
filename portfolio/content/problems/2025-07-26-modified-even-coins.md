---
title: Modified Even Coins
subtitle: 
author: 
date: '2025-07-26'
tags:
  - Probability
  - Conditional Probability
  - Law of Total Probability
---

Problem:
>`$n$` coins are laid out in front of you. 
>
>One of the coins is fair, while the other `$n-1$` have probability `$0 < \lambda < 1$` of showing heads. 
>
>If all `$n$` coins are flipped, find the probability of an even amount of heads.

---

Let `$\mathcal{F}$` represent the outcome of the fair coin and `$\mathcal{H}$` the total heads from the `$n-1$` unfair coins. 

This means the total number of heads is given by `$\mathcal{F} + \mathcal{H}$` and we are specifically interested in the probability that `$\mathcal{F} + \mathcal{H}$` is even. 

Now if we condition on the fair coin (assuming that `$\mathcal{F} + \mathcal{H}$` is even):

`$$P(\mathcal{F} + \mathcal{H}) = P(\mathcal{F} + \mathcal{H} \mid \mathcal{F} = 0)P(\mathcal{F} = 0) + P(\mathcal{F} + \mathcal{H} \mid \mathcal{F} = 1)P(\mathcal{F} = 1)$$`

We know that the probability of the outcomes for the fair coin are:

`$$P(\mathcal{F} = 0) = P(\mathcal{F} = 1) = 0.5$$`

and we also know the outcomes of `$\mathcal{F}$` is either 0 or 1, so we can simplify the above to:

`$$P(\mathcal{F} + \mathcal{H}) = P(0 + \mathcal{H}) \cdot 0.5 + P(1 + \mathcal{H}) \cdot 0.5$$`

In this case where `$\mathcal{F} = 0$` we need `$\mathcal{H}$` to be even and for `$\mathcal{F} = 1$` we need `$\mathcal{H}$` to be odd. 

This then leaves us with:

`$$P(\mathcal{F} + \mathcal{H}) = 0.5 \cdot P(\mathcal{H} \text{ is even}) + 0.5 \cdot P(\mathcal{H} \text{ is odd})$$`

Because of the law of total probability both cases of `$P(\mathcal{H} \text{ is even})$` and `$P(\mathcal{H} \text{ is odd})$` must sum to 1. This means `$P(\mathcal{H} \text{ is odd})$` is equivalent to `$1 - P(\mathcal{H} \text{ is even})$`.

If we put this back into the equation we have:

`$$\begin{align*}
P(\mathcal{F} + \mathcal{H} \text{ is even}) &= 0.5 \cdot P(\mathcal{H} \text{ is even}) + 0.5 \cdot (1 - P(\mathcal{H} \text{ is even})) \\
&= 0.5 \cdot P(\mathcal{H} \text{ is even}) + 0.5 \cdot 1 - 0.5 \cdot P(\mathcal{H} \text{ is even}) \\
&= 0.5 \cdot P(\mathcal{H} \text{ is even}) + 0.5 - 0.5 \cdot P(\mathcal{H} \text{ is even}) \\
&= [0.5 \cdot P(\mathcal{H} \text{ is even}) - 0.5 \cdot P(\mathcal{H} \text{ is even})] + 0.5 \\
&= 0 + 0.5 \\
&= 0.5
\end{align*}$$`

Which means the probability of an even number of heads is 0.5.




