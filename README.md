# KuhnTrainer
An implementation of the CounterFactual Regret Minimisation (CFRM) algorithm for solving 2 and 3-player Kuhn Poker. This project was developed by me as part of my 3rd Year courses in Computer Science at the University of Manchester.

CFRM is a machine learning algorithm developed to solve games with hidden information. A game is considered to have 'hidden information' when a player cannot access information held by an opponent. One example is poker, where players cannot see opponent's cards. This program focuses on solving a simplified version called Kuhn Poker with 2 and 3 players.

# Running the program
This program works by running the CFRM algorithm repeatedly on a Kuhn Poker game to determine the optimal strategy for each player (the epsilon-Nash equilibrium) as closely as possible. There are several settings options available here:

- *Game Type* - Select the 2 or 3-player version of the game.
- *Number of trials* - Sets the number of times to repeat the learning process for the given game. Output results will be averaged over the number of trials selected.
- *Number of iterations* - Sets the number of times to run the CFRM algorithm.
- *Number of samples* - Sets the number of samples to select from output results to use for displaying graphs.
- *X-axis units* - Sets the units used for the x-axis of displayed graphs. Available units are number of iterations, game tree nodes reached and calculation times.

Once the learning process is complete, the program outputs the following results:

- The calculated pass/bet probabilities for each game tree node.
- The calculated and expected game values for each player, as well as their difference.
- The epsilon-Nash equilibrium value, total game nodes reached and average calculation time per trial.
- Graphs displaying the average game value for each player and the average epsilon-Nash value. The 2-player version also includes graphs of average betting probability for each tree node, though these are omitted in the 3-player version.

# Testing
The game includes tests for both game versions. The tests run each game for ten million iterations and compares the calculated bet probabilities for each game tree node against the expected value. The bet values for 2 and 3-player are checked within a tolerance of 0.01 and 0.03 respectively.
