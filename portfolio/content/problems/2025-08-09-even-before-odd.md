---
title: Even Before Odd
subtitle: 
author: 
date: '2025-08-09'
tags:
  - Probability
  - Combinatorics
---

Problem:

> Suppose you roll a fair 6-sided die until you've seen all 6 faces.
> 
> What is the probability you won't see an odd numbered face until you have seen all even numbered faces?

Since this problem is symmetric, it makes it much simpler to solve. This means if the question was flipped, the answer would be the same.

All we need to do is count the number of possible sequences where the first 3 unique faces seen are even, then divide by the total number of possible orderings.

```python
import math

print((math.factorial(3) * math.factorial(3)) / math.factorial(6))
```

Which gives us a probability of 0.05.

If you wanted to solve this problem differently, you could also use `itertools` and generate all possible sequences of rolls and then count the number of sequences where the first 3 unique faces seen are even.

```python
import itertools

sequences = itertools.permutations([1, 2, 3, 4, 5, 6], r=6)

valid = 0
total = 0

for sequence in sequences:
    if all(i in {2, 4, 6} for i in sequence[:3]):
        valid += 1
    total += 1
    
print(valid / total)
```

```python
0.05
```

But of course the factorial approach is much quicker.