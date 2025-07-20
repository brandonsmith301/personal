---
title: Greedy Pirates
subtitle: 
author: 
date: '2025-07-20'
tags:
  - Game Theory
  - Backward Induction
---

Problem:

> A treasure of 100 gold coins must be divided among five pirates, ranked by seniority from 1 (least senior) to 5 (most senior). They follow these rules to split the coins:
>
> 1. The most senior pirate proposes how many coins to distribute to each pirate.
> 2. All pirates vote on the proposal, including the pirate that made the proposal. If 50% or more of the pirates accept the proposal, the coins are distributed as proposed.
> 3. Otherwise, if more than 50% of the pirates reject the proposal, the most senior pirate is tossed overboard. The process restarts with the next most senior pirate making a proposal and the remaining pirates casting votes.
> 4. This process repeats until a proposal is accepted or only one pirate remains, in which case the last pirate gets all 100 gold coins.
>
> Pirates prioritize survival over wealth. The pirates will act rationally to maximize their gain while ensuring they survive. How many coins does the most senior pirate (pirate 5) get if he makes an optimal proposal? If the most senior pirate is always thrown overboard, answer 0.

---

This is my first attempt at a game theory problem so this is a bit of a learning experience. The specific algorithm which I am going to use is called ***backward induction***.

The definition of backward induction is:

> ... corresponds to the assumption that it is common knowledge that each player will act rationally at each future node where he moves — even if his rationality would imply that such a node will not be reached.

The resource which I got this definition from can be found [here](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/4b4412575dc74593c9d9c59e94427b69_MIT14_12F12_chapter9.pdf).

The idea is to solve the problem backwards, starting with the last pirate and working our way back to the first pirate.

Before we start, it is helpful to establish some facts about the problem.

There are 5 pirates, ranked by seniority from 1 (least senior) to 5 (most senior). Each pirate `$i$` seeks to maximise `$x_i$` subject to:

`$$\begin{align*}
\sum_{j=1}^k x_j &= 100\\
x_j &\geq 0 \quad \text{for all } j \in \{1,\ldots,k\}\\
S_i &= 1 
\end{align*}$$`

where `$k$` is the number of pirates remaining in the current subgame and `$S_i = 1$` indicates that pirate `$i$` survives.

The rules are that pirate `$k$` proposes `$(x_1, \ldots, x_k)$` and if `$\sum v_j \geq \lceil \frac{k}{2} \rceil$` (where `$v_j \in \{0, 1\}$` is the vote of pirate `$j$`) then distribute coins as proposed, otherwise eliminate pirate `$k$` and set `$k \leftarrow k-1$`.

The process is repeated until a proposal is accepted or only one pirate remains, in which case the last pirate gets all 100 gold coins.

---

#### Problem: `$k=1$`

In this case, there is only one pirate left, so the pirate gets all 100 gold coins.

#### Problem: `$k=2$`

Since there are only two pirates there is no risk of being thrown overboard, the most senior pirate `$i_2$` will obviously propose `$x_1 = 0$` and `$x_2 = 100$`. 

#### Problem: `$k=3$`

We know from `$k=2$` that `$i_2$` will get all 100 gold coins, this means that `$i_3$` can give `$i_1$` a single gold coin and keep 99 gold coins for himself. 

#### Problem: `$k=4$`

There are now four pirates, and knowing that `$i_3$` will give `$i_1$` a single gold coin when `$k=3$` and `$i_2$` will then get nothing, we can see that `$i_4$` will give `$i_2$` a single gold coin and keep 99 gold coins for himself.

#### Problem: `$k=5$`

In order for `$i_5$` to get the most gold coins and survive, he must give both `$i_3$` and `$i_1$` a single gold coin and keep 98 gold coins for himself since they both get nothing when `$k=4$`.

---

The most senior pirate `$i_5$` will give both `$i_3$` and `$i_1$` a single gold coin and keep 98 gold coins for himself.






