# Ultimate Tic-Tac-Toe: Adversarial Search Agent

## Overview
This repository contains a series of progressively optimized AI agents designed to play **Ultimate Tic-Tac-Toe**, a complex variant of the classic game played on a 3x3 grid of 3x3 boards. The project demonstrates the implementation and optimization of adversarial search algorithms, strictly adhering to a **2.5-second time limit** per move.

## The Challenge
Ultimate Tic-Tac-Toe is a fully observable, strategic, and deterministic game. The complexity arises from the **"send rule"**: a player's move in a local 3x3 board dictates which local board the opponent must play in next. This drastically increases the branching factor compared to standard Tic-Tac-Toe, requiring highly efficient search space exploration.

### Constraints
* **No Monte Carlo Tree Search (MCTS):** Pure adversarial search only.
* **No Bitboard State Modifications:** Standard state representation required.
* **Strict Timing:** 2.5s execution limit per move.

---

## Agent Evolution & Algorithmic Optimizations
To handle the massive state space, I iteratively developed 10 agents, grouped into four main architectural phases:

### Phase A: The Baseline (Minimax)
* Implemented a standard **Minimax** algorithm to explore the game tree.
* Established a baseline heuristic evaluation function, primarily checking for terminal states and naive local board control.

### Phase B: Pruning the Tree (Alpha-Beta Pruning)
* Upgraded the search using **Alpha-Beta Pruning** to eliminate branches that mathematically cannot influence the final decision.
* **Heuristic Upgrade:** Designed a more sophisticated evaluation function that weighs center-board dominance heavily (multiplying value by 1.3) and scans for local and meta-board "two-in-a-row" threats.

### Phase C: State Hashing & Symmetries (Transposition Tables)
* **Transposition Tables (Memoization):** Implemented a caching system to store previously evaluated board states, preventing redundant calculations.
* **Symmetry Reduction:** Since the board is symmetrical, I engineered a 9x9 matrix transformation to hash and recognize rotated (90°, 180°, 270°) and horizontally flipped board states. This drastically reduced the search space.

### Phase D: Search Prioritization (Move Ordering)
* Integrated **Move Ordering** to maximize the efficiency of Alpha-Beta pruning.
* By sorting the available actions based on a "quick evaluation" of the resulting state before diving into the recursive search, the algorithm finds optimal cutoffs much faster, allowing it to search deeper within the 2.5-second threshold.

---

## Results & Reflections
The final agent (**Agent D3**) successfully navigates the extreme branching factor of Ultimate Tic-Tac-Toe using pure adversarial search techniques. While the highest-tier benchmark testing required machine learning heuristics to pass, this pure-search approach successfully defeats a wide array of sophisticated agents by maximizing tree depth through rigorous algorithmic optimization.
