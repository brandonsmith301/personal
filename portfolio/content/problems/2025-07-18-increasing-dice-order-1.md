---
title: Increasing Dice Order I
subtitle: 
author: 
date: '2025-07-18'
tags:
  - Probability
  - Combinatorics
---


Problem:

> You roll a fair `$6$`-sided die twice. Calculate the probability that the value of the first roll is strictly less than the value of the second roll.

---

There are six equal cases where `$(i,i)$` for `$i = 1, 2, \ldots, 6$`, which leaves us with 30 remaining outcomes.

Since both dice are identical, the number of outcomes where the first die is greater than the second equals the number where the first is less than the second. Therefore, each case has `$\frac{30}{2} = 15$` outcomes.

The probability that the first die is less than the second is `$\frac{15}{36} = \frac{5}{12} \approx 0.417$`.
