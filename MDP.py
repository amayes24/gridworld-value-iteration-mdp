## DO NOT MAKE ANY CHANGES TO THE CODE UNLESS INSTRUCTED. ONLY MODULES STATED IN THE INSTRUCTION CAN BE IMPORTED
import tkinter as tk
##Import necessary modules for your project
import math

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}
CELL_SIZE = 80
FONT_VALUE = ('Arial', 10)
FONT_LABEL = ('Arial', 16, 'bold')

def parse_gridworld(filename): # This function will extract all the information from the gridworld spec and design the griworld.
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    idx = 0
    rows, cols = map(int, lines[idx].split()); idx += 1
    start = tuple(map(int, lines[idx].split())); idx += 1

    terminals = {}
    if lines[idx] == 'TERMINALS':
        idx += 1
        while lines[idx] != 'END':
            r, c, rew = lines[idx].split()
            terminals[(int(r), int(c))] = float(rew)
            idx += 1
        idx += 1

    walls = set()
    if lines[idx] == 'WALLS':
        idx += 1
        while lines[idx] != 'END':
            r, c = map(int, lines[idx].split())
            walls.add((r, c))
            idx += 1
        idx += 1

    step_reward = float(lines[idx]); idx += 1
    noise = float(lines[idx]); idx += 1

    return rows, cols, start, terminals, walls, step_reward, noise

def move(state, action, rows, cols, walls):
    r, c = state
    if action == 'UP': r -= 1
    if action == 'DOWN': r += 1
    if action == 'LEFT': c -= 1
    if action == 'RIGHT': c += 1

    if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in walls:
        return state
    return (r, c)


def get_transitions(state, action, rows, cols, walls, noise):
    """Returns list of (probability, next_state)"""
    if noise == 0.0:
        return [(1.0, move(state, action, rows, cols, walls))]

    if action in ['UP', 'DOWN']:
        alt = ['LEFT', 'RIGHT']
    else:
        alt = ['UP', 'DOWN']

    transitions = []
    transitions.append((1 - noise, move(state, action, rows, cols, walls)))
    transitions.append((noise / 2, move(state, alt[0], rows, cols, walls)))
    transitions.append((noise / 2, move(state, alt[1], rows, cols, walls)))
    return transitions

# ==================================================
# Value Iteration
# ==================================================

def value_iteration(rows, cols, terminals, walls, step_reward, noise, gamma, epsilon=1e-4):
    # set up values for every valid state
    V = {}

    for r in range(rows):
        for c in range(cols):
            s = (r, c)

            if s in walls:
                continue
            elif s in terminals:
                V[s] = terminals[s]
            else:
                V[s] = 0.0

    iterations = 0

    while True:
        delta = 0
        new_V = V.copy()
        iterations += 1

        for r in range(rows):
            for c in range(cols):
                s = (r, c)

                if s in walls:
                    continue

                if s in terminals:
                    new_V[s] = terminals[s]
                    continue

                best_value = -math.inf

                for action in ACTIONS:
                    q_value = 0

                    for prob, next_state in get_transitions(s, action, rows, cols, walls, noise):
                        reward = terminals.get(next_state, step_reward)
                        q_value += prob * (reward + gamma * V[next_state])

                    if q_value > best_value:
                        best_value = q_value

                new_V[s] = best_value
                delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < epsilon:
            break

    return V, iterations

def init_values(rows, cols, walls, terminals):
    """Initialize all state values to 0.0 except walls."""
    values = {}
    pol = {}

    for r in range(rows):
        for c in range(cols):
            s = (r, c)

            if s in walls:
                continue
            elif s in terminals:
                values[s] = terminals[s]
            else:
                values[s] = 0.0
                pol[s] = 'UP'

    return values, pol

# ==================================================
# Policy Extraction
# ==================================================

def extract_policy(V, rows, cols, terminals, walls, step_reward, noise, gamma):
    policy = {}

    for r in range(rows):
        for c in range(cols):
            s = (r, c)

            if s in terminals or s in walls:
                continue

            best_a, best_q = None, -math.inf

            for a in ACTIONS:
                q = 0

                for p, s2 in get_transitions(s, a, rows, cols, walls, noise):
                    reward = terminals.get(s2, step_reward)
                    q += p * (reward + gamma * V[s2])

                if q > best_q:
                    best_q, best_a = q, a

            policy[s] = best_a

    return policy

def draw_grid(rows, cols, start, terminals, walls, values, policy, title='Grid World with State Values'):
    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(root, width=cols * CELL_SIZE, height=rows * CELL_SIZE)
    canvas.pack()

    for r in range(rows):
        for c in range(cols):
            x1 = c * CELL_SIZE
            y1 = r * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            fill = 'white'
            if (r, c) in walls:
                fill = 'black'
            elif (r, c) in terminals:
                fill = 'lightgreen' if terminals[(r, c)] > 0 else 'lightcoral'

            canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline='gray')

            # Start state
            if (r, c) == start:
                canvas.create_text(x1 + CELL_SIZE / 2, y1 + 14, text='S', font=FONT_LABEL)

            # Terminal reward
            if (r, c) in terminals:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE / 2,
                    text=f'{terminals[(r, c)]:.2f}',
                    font=('Arial', 14, 'bold'),
                    fill='black'
                )

            # State value
            if (r, c) in values and (r, c) not in terminals:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE - 18,
                    text=f'V={values[(r, c)]:.2f}',
                    font=('Arial', 11, 'bold'),
                    fill='blue'
                )

            # Policy arrow
            if (r, c) in policy:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE / 2,
                    text=ARROWS[policy[(r, c)]],
                    font=('Arial', 22, 'bold'),
                    fill='black'
                )

    root.mainloop()


if __name__ == '__main__':
    #Provide the gridworld file name as input
    print("Enter the gridworld spec")
    grid = input()

    rows, cols, start, terminals, walls, step_reward, noise = parse_gridworld(grid)

    gamma = [0.9, 0.95, 0.99]

    for discount in gamma:
        V, iterations = value_iteration(rows, cols, terminals, walls, step_reward, noise, discount)
        policy = extract_policy(V, rows, cols, terminals, walls, step_reward, noise, discount)

        print("--------------------------------")
        print("Gamma:", discount)
        print("Iterations until convergence:", iterations)
        print("Final policy:")

        for r in range(rows):
            row_policy = []
            for c in range(cols):
                s = (r, c)

                if s in walls:
                    row_policy.append("W")
                elif s in terminals:
                    row_policy.append("T")
                else:
                    row_policy.append(ARROWS[policy[s]])

            print(row_policy)

        draw_grid(
            rows,
            cols,
            start,
            terminals,
            walls,
            V,
            policy,
            title=f'Grid World Values and Policy - gamma={discount}'
        )
