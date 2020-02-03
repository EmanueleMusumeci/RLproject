# Reinforcement Learning - TRPO - Tensorflow 2

Small homemade modular framework created as a University project, to implement and test Reinforcement Learning algorithms, with a working implementation of [TRPO](https://arxiv.org/pdf/1502.05477.pdf) 

## Basic instructions

### Prerequisites

I recommend using a Linux distro (Ubuntu mainly) and [creating a Python virtual environment](https://docs.python.org/3/library/venv.html), then [activate it](https://stackoverflow.com/questions/14604699/how-to-activate-virtualenv).

Requirements:

* OpenAI gym

* Tensorflow 2

* Python OpenCV2 (for Atari environments image preprocessing)

* Numpy

### Installing

Finally, clone this repo:

```
git clone https://github.com/EmanueleMusumeci/RLproject
```
### Using

The algorithm module (Example: TRPOAgent.py) is executable. Run it as:
```
python TRPOAgent.py <environment_name> train -h
```
to receive additional help about training the agent on a certain environment, or
```
python TRPOAgent.py <environment_name> run -h
```
to receive help about running the trained agent.

### Credits

* [TRPO paper](https://arxiv.org/pdf/1502.05477.pdf)
* These two awesome Medium articles about TRPO:
    * [RL — Trust Region Policy Optimization (TRPO) Explained](https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9)
    * [RL — Trust Region Policy Optimization (TRPO) Part 2](https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a)
* [OpenAI SpinningUp documentation](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
* All the awesome implementations here on GitHub I used to understand the TRPO algorithm