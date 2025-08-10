---
title: Dot Placement
subtitle: 
author: 
date: '2025-08-09'
tags:
  - Probability
  - Combinatorics
---

Problem:

> You place three dots along the edges of an octagon at random. What is the probability that all three dots lie on distinct edges of the octagon?

This is a very easy problem, which we can solve computationally:

```python
import itertools

generator = itertools.product([1,2,3,4,5,6,7,8], repeat=3)

valid = 0
total = 0
for i in generator:
    if i[0] != i[1] and i[1] != i[2] and i[0] != i[2]:
        valid += 1
    total += 1
    
print(valid / total)
```

```python
0.65625
```

This gives us a probability of 0.65625.

If we want to do this mathematically, we first need to count the total number of ways to place 3 dots along the edges of the octagon which is `$8^3 = 512$` since for every dot we have 8 choices each time.

Then we need to count the number of ways to place 3 dots along the edges of the octagon where all 3 dots are on distinct edges which is `$8 \cdot 7 \cdot 6 = 336$` since for the first dot we have 8 choices, for the second dot we have 7 choices, and for the third dot we have 6 choices.

Therefore, the probability is `$\frac{336}{512} = 0.65625$`.