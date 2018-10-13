# Pytorch-Policy-Gradient
 
Pytorch Implementation for Policy Gradient Methods

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
    # By default use PPO
    python run.py --batch_size=64 --rllr=1e-3 --env HalfCheetah-v1 --episodes 8 --shuffle --entropy_para = 0
```

## Note

For current TRPO  implementation, it's better to use seperate network for policy and value. Using share-parameter network would make training unstable.