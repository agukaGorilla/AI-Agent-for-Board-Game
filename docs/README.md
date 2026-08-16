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

## ⚠️ Note on Code Authorship & AI Usage

Transparency is important to me. Please note the following regarding the architecture of this repository:

*   **Core Algorithms (`My Agents/`):** The 10 distinct AI agents (Agent A through Agent D3), their underlying heuristic math, alpha-beta pruning logic, and transposition tables are **100% my original work**. This project was developed as university coursework in 2025, where it earned a final score of **9.5 / 10**.
*   **Repository Timeline:** You may notice that the commit history for this repository is very recent. During my studies, university policy strictly prohibited making coursework public. I have only recently uploaded this code to GitHub to share it with prospective employers.
*   **Visual Engine & Mechanics (`game_engine/`):** To make my core algorithms easy to evaluate, I recently used AI assistance to generate the CLI visualizer, ANSI color rendering, and game loop mechanics found in `main.py` and `utils.py`. These files were created specifically to wrap my raw algorithmic logic into a clean, interactive testing environment so technical interviewers and recruiters can easily visualize and play against the agents.


## Gameplay Showcase

<p align="center">
  <b>Interactive Agent Selection</b><br>
  <i>Menu allowing users to select human play or deploy one of the 10 custom AI agents.</i><br>
  <img src="Screenshot 2026-08-16 204850.png" alt="Agent Selection Menu" width="80%">
</p>

<br>

<p align="center">
  <b>Input Validation & Game Rules</b><br>
  <i>Human vs. AI gameplay demonstrating strict input validation and "send rule" enforcement.</i><br>
  <img src="Screenshot 2026-08-16 204829.png" alt="Rule Enforcement and Input Error" width="80%">
</p>

<br>

<p align="center">
  <b>Algorithmic Battles</b><br>
  <i>Mid-game tactical evaluation as two algorithmic agents battle for meta-board control.</i><br>
  <img src="Screenshot 2026-08-16 205032.png" alt="Mid-game AI vs AI" width="80%">
</p>

<br>

<p align="center">
  <b>Dynamic Visual Rendering</b><br>
  <i>ANSI color rendering displaying local board captures during an intense matchup.</i><br>
  <img src="Screenshot 2026-08-16 202313.png" alt="Board Captures Rendering" width="80%">
</p>

<br>

<p align="center">
  <b>Terminal Victory State</b><br>
  <i>Final state showing a decisive victory for Player 1.</i><br>
  <img src="Screenshot 2026-08-16 205053.png" alt="Victory Screen" width="80%">
</p>

### Prerequisites
* **Python 3.8+**
* **NumPy:** Required for multidimensional state management.
  ```bash
  pip install numpy

### How to Run the Game

Navigate to the game engine directory and execute the main Python file to start the interactive CLI:

```bash
cd game_engine
python main.py