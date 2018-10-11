# Pytorch Implementation for Policy Gradient Methods

This repo contains several policy gradient algorithm implementation.

Environments with continuous / discrete action space are both supported.

Support MLP / Conv Policy.

## Currently contains:
* Basic Reinforce
* Actor-Critic
* TRPO
* PPO
* A3C

## TODO:
1. Add More Algorithm
* Soft-actor-critic
* DDPG
* ...

## Basic Usage
```python
    python run.py --agent=ppo --batch_size=512 --rllr=1e-3 --env=PongNoFrameskip-v4
```
