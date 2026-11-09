# Adaptive Prompt Selector for LLM Auditing

## Overview
This project implements a Reinforcement Learning framework to systematically audit Large Language Models for training data memorization. It uses an RL agent to learn which extraction prompts are most effective at forcing an LLM to leak its training data, operating under a strict prompt-budget constraints.

## Methodology & Hardware Optimization
Running continuous LLM generation during an active RL training loop is highly memory-intensive and creates massive bottlenecks on constrained hardware (e.g., 8GB unified memory). 

To ensure optimal performance and rapid iteration without relying on expensive cloud GPUs, this framework implements a **Mathematical LLM Simulation (MVP)**:
* Rather than loading a full LLM into local memory to evaluate every single step, the Gymnasium environment assigns statistical success probabilities to various prompt templates based on empirical vulnerability rates. 
* The environment evaluates the chosen action (prompt) against these probabilities to simulate the LLM's response.
* This allows the REINFORCE algorithm to train the agent in seconds via Apple Metal Performance Shaders (`mps`), validating the RL methodology and policy network architecture perfectly without hardware overflow.

## Environment Architecture (MDP)
* **State Space:** Continuous array `[current_file_index, attempts_used]`, normalized between 0 and 1.
* **Action Space:** Discrete space of 5 possible prompt strategies.
* **Reward Function:** 
  * `+10` for successfully extracting memorized data (Terminal).
  * `-1` for a failed attempt (Non-terminal).
* **Constraints:** Strict budget of 5 attempts per episode.

## Project Structure
* `dataset_builder.py`: Extracts and processes a subset of Python code from `bigcode/the-stack-v2` for evaluation.
* `llm_env.py`: Custom Gymnasium environment defining the MDP and simulating LLM extraction probabilities.
* `train_agent.py`: PyTorch implementation of the REINFORCE algorithm to train the policy network.
* `adaptive_prompt_policy.pth`: The saved weights of the fully trained RL agent.