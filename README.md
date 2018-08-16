# Pytorch Implementation for Policy Gradient Methods

## Currently contains:
* Basic Reinforce
* Actor-Critic
* TRPO
* PPO
* A3C

P.S: Currently support discrete action space

## TODO:
1. Add More Algorithm
* Soft-actor-critic
* DDPG
* ...
2. Add continuous support

## Basic Usage
```python
    python run.py --agent=ppo --batch_size=512 --rllr=1e-3 --env=PongNoFrameskip-v4
```
