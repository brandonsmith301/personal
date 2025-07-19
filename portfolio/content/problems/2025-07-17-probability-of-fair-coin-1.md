---
title: Probability of Unfair Coin I
subtitle: 
author: 
date: '2025-07-17'
tags:
  - Probability
  - Conditional Probability
  - Bayes' Theorem
---


Problem:

> You have a pile of 100 coins. 1 of the coins is an unfair coin and has heads on both sides. The remaining 99 coins are fair coins. 
>
>You randomly select a coin from the pile and flip it 10 times. The coin lands heads all 10 times. 
>
>Calculate the probability that the coin you selected is the unfair coin.

---

The first thing to do is to set up a Bayes' table, which we will do so using Pandas (remember to `import pandas as pd`) if you are following along.

```python
table = pd.DataFrame(index=["fair", "unfair"])
```

The reason we set our index to `["fair", "unfair"]` is because we have two possible outcomes. We can then set the prior to the probability of selecting a fair or unfair coin.

```python
table["prior"] = 99/100, 1/100
```

The likelihood is going to be the probability of getting 10 heads with a fair or unfair coin. Now the probability of getting 10 heads with a fair coin is `$(0.5)^{10}$` and the probability of getting 10 heads with an unfair coin is `$1$`, which means:

```python
table["likelihood"] = (0.5)**10, 1
```

Then we do our update step, which is to multiply the prior by the likelihood.

```python
table["unnorm"] = table["prior"] * table["likelihood"]
table["posterior"] = table["unnorm"] / table["unnorm"].sum()
```

If we show the posterior probability for both fair and unfair coins, we get:

```python
fair      0.088157
unfair    0.911843
```

Therefore, the probability that the coin you selected is the unfair coin is `$\approx 0.912$`.