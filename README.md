# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Access the unwrapped environment to use the transition model
env = env.unwrapped

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of States :", n_states)
print("Number of Actions:", n_actions)

# Parameters
gamma = 0.99
theta = 1e-8

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions

# Initialize value function
V = np.zeros(n_states)

# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Performs iterative policy evaluation using the
    Bellman Expectation Equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V          : State-value function
        iteration  : Number of iterations
    """

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    P = env.P

    V = np.zeros(n_states)

    iteration = 0

    while True:

        delta = 0

        for state in range(n_states):

            old_value = V[state]

            new_value = 0

            # Bellman Expectation Equation
            for action in range(n_actions):

                action_probability = policy[state][action]

                for probability, next_state, reward, done in P[state][action]:

                    new_value += (
                        action_probability
                        * probability
                        * (reward + gamma * V[next_state])
                    )

            V[state] = new_value

            delta = max(delta, abs(old_value - new_value))

        iteration += 1

        if delta < theta:
            break

    return V, iteration

# Run policy evaluation

V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name:Bhuvaneshwaran H")
print("Register Number:212223240018")
print("Number of Iterations:", iterations)

print("\nState-Value Function:\n")
print(np.round(V, 4))

print("\nState-Value Function as 4x4 Grid:\n")
print(np.round(V.reshape((4, 4)), 4))

env.close()


```

---

## Output

```text


Number of States : 16
Number of Actions: 4

Name:Bhuvaneshwaran H
Register Number:212223240018
Number of Iterations: 54

State-Value Function:

[0.0124 0.0104 0.0193 0.0095 0.0148 0.     0.0389 0.     0.0326 0.0843
 0.1378 0.     0.     0.1703 0.4336 0.    ]

State-Value Function as 4x4 Grid:

[[0.0124 0.0104 0.0193 0.0095]
 [0.0148 0.     0.0389 0.    ]
 [0.0326 0.0843 0.1378 0.    ]
 [0.     0.1703 0.4336 0.    ]]


```
---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text



```
for gamma = 0.84


Name:Bhuvaneshwaran H
Register Number:212223240018
Number of Iterations: 32

State-Value Function:

[0.0024 0.0024 0.0068 0.0025 0.0041 0.     0.0207 0.     0.0131 0.0452
 0.0915 0.     0.     0.1104 0.3701 0.    ]

State-Value Function as 4x4 Grid:

[[0.0024 0.0024 0.0068 0.0025]
 [0.0041 0.     0.0207 0.    ]
 [0.0131 0.0452 0.0915 0.    ]
 [0.     0.1104 0.3701 0.    ]]



---


