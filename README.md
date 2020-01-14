# gym-nav2d
A simple continuous action 2d navigation task that uses the gym interface.

## How to use

1. Clone this repository and navigate into this folder
2. Activate your python environment and run `pip install -e .` to build a package and link it locally to folder.
3. Import gym, see test/RandomAgent for a basic usage.

## Overview
### Details
* Names: 
    * gym_nav2d:nav2dVeryEasy-v0
    * gym_nav2d:nav2dEasy-v0
    * gym_nav2d:nav2dHard-v0
    * gym_nav2d:nav2dVeryHard-v0
* Category: Classic Control

### Description
Find a point goal in a simple 2D navigation maze.

## Author
Florian Soulier

## Observation
The observation depends on the gym you load. _gym_nav2d:nav2dVeryEasy-v0_, _gym_nav2d:nav2dEasy-v0_, _gym_nav2d:nav2dHard-v0_ have the following observation:

**Hint**: The environment works with a two dimensional grid from 0..255 but we convert these values into the interval -1..1. The observation is of type Box[float, float, float, float, float].

Type: Box(5)

| Num  | Observation                     | Min  | Max    |
| ---- | ------------------------------- | ---- | ------ |
| 0    | agent position x                | -1    | 1    |
| 1    | agent position y                | -1    | 1    |
| 2    | goal position x                 | -1    | 1    |
| 3    | goal position y                 | -1    | 1    |
| 4    | distance between agent and goal | 0    | 1    |

**4** is calculated by: $$\sqrt{(255-0)^2+(255-0)^2}/362,22$$

For _gym_nav2d:nav2dVeryHard-v0_ only **4** is given. 

## Actions

Type: Box(2)

| Num  | Action    | Min  | Max  |
| ---- | --------- | ---- | ---- |
| 0    | degree    | -1    | 1  |
| 1    | step_size | -1    | 1   |

**Hint**: We expect floats between -1 and 1 (e.g. output of tanh), in form of Box[float, float] and convert those to an angle in rad between 0 and pi and a steps size between 0 and 10.

## Reward

step reward is calculated by following formula:

$$R_t = - d/10 - 1$$

the step reward is actual distance and -1 per timestep.

the cumulated reward is:

$$R = \sum R_t$$

If the agent reaches the goal, it gets a bonus reward of 1000. 

## Starting state

The starting state depends on the environment. Here is an overview:

| Name     | start    | goal     | observation |
| -------- | -------- | -------- | ----------- |
| VeryEasy | fixed    | fixed    | 0..4        |
| Easy     | variable | fixed    | 0..4        |
| Hard     | variable | variable | 0..4        |
| VeryHard | variable | variable | only 4      |

## Episode Termination

The episode ends when you  reach eps-position (see *gym_nav2d:nav2dVeryEasy-v0* init method ) or if 100 iterations are reached.

## Solved Requirements

Well, in the VeryEasy env an agents needs at maximum $$ceil(\sqrt{(200-10)^2+(200-10)^2}:10) = 27$$ steps. You can poof this with the OracleAgent in test folder. This env is no big challenge and meant to be a test env, wether your RL agent can learn anything.

The Easy case is meant to be a common RL problem, where you always have the same goal but your agent starts at different positions.

The Hard env is a lot harder, since the goal will be variable too and will take a lot more training time as the Easy env. This can be used to compare with meta learners.

The VeryHard env only gives the distance as observation and might not be solvable for a common RL agent. If your agent cannot learn to solve this environment, do not waste your time in trying it.

In the Easy, Hard and VeryHard env, a very good agents needs max $$ceil(\sqrt{(255-0)^2+(255-0)^2}:10) = 37 $$ steps in the worst case. Since they distinct in start, goal positions and observations, you can assume more steps also as solved, especially Hard and VeryHard.

## Agents

Two sample agents are implemented. RandomAgents mimics a completely untrained agent and OracleAgent mimics an agent, that has learned a strong policy.


RandomAgent             | OrcaleAgent 
:-------------------------:|:-------------------------:
![random_agent](README.assets/random_agent.png)|  ![oracle_agent](README.assets/oracle_agent.png)
