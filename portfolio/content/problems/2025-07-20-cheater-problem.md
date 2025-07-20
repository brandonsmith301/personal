---
title: Guessing the answer
subtitle: 
author: 
date: '2025-07-20'
tags:
  - Probability
  - Conditional Probability
  - Bayes' Theorem
---

Problem:

> As a good test taker, on a multiple choice exam with 5 options per question, Gabe either knows the answer to a question beforehand or chooses an answer completely at random. 
>
>If he knows the answer beforehand, he selects the correct answer.
>
>The probability that Gabe knows the answer to any given question is 0.6. Find the probability that an answer that was correct was one for which he knew the answer.

---

There are two possible events, Gabe either knows the answer or he doesn't. 

```python
table = pd.DataFrame(index=["Knows", "Doesn't know"])
```

Using the information provided in the problem, we can fill in the table as follows:

```python
table["prior"] = 0.6, 0.4
```

Now the likelihood is the probability of getting the correct answer given that Gabe knows the answer or doesn't know the answer. 

```python
table["likelihood"] = 1, 1/5
```

Then we can work out the posterior probability of Gabe knowing the answer given that he got the answer correct.

```python
table["posterior"] = table["unnorm"] / table["unnorm"].sum()
```

The probability of Gabe knowing the answer given that he got the answer correct is `$\approx 0.88$`. The full working table is shown below:

| Event | Prior | Likelihood | Unnorm | Posterior |
|-------|------------|--------|-----------|-----------|
| Knows | 0.6        | 1.0    | 0.60      | 0.88  |
| Doesn't know | 0.4        | 0.2    | 0.08      | 0.12  |





