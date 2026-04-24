Project: Value Iteration in Grid World

This project implements the Value Iteration algorithm for solving a Grid World Markov Decision Process (MDP). The program computes the state values V(s), extracts the optimal policy π(s), and displays both on a GUI.

How to Run:

1. Open terminal and navigate to the project folder:
   cd Desktop/AI_Project_2

2. Run the program:
   python3 MDP.py

3. When prompted, enter the grid file name:
   gridworld_easy.txt
   gridworld_medium.txt
   or gridworld_hard.txt

4. The program will:
   - Run value iteration for gamma = 0.90, 0.95, 0.99
   - Print the number of iterations until convergence
   - Display the final state values and policy on the GUI

5. Close each GUI window to move to the next gamma value.

Notes:

- Walls are shown in black, terminal states are colored (green for +1 and red for -1).
- Arrows represent the optimal policy at each state.
- State values V(s) are shown in blue inside each cell.
- Screenshots were taken after convergence for each grid and discount factor.

Observations:

I ran value iteration using gamma values 0.90, 0.95, and 0.99 on the grid worlds. One thing I noticed is that as gamma increases, the number of iterations to converge also increases. For example, the medium and hard grids took more iterations for higher gamma values.

The state values also increase as gamma increases. This makes sense because a higher gamma puts more weight on future rewards, so the values get larger overall.

The policy mostly stays similar across different gamma values. In all cases, the agent moves toward the +1 terminal state and avoids the -1 state. In the harder grid, I noticed that with higher gamma, the agent is slightly more careful and avoids risky paths near the negative reward.

Overall, gamma affects how much the agent cares about future rewards, but the general path to the goal stays mostly the same.