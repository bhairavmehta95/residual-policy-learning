# Residual Policy Learning
## Tom Silver*, Kelsey Allen*, Josh Tenenbaum, Leslie Kaelbling

## Abstract
We present Residual Policy Learning (RPL): a simple method for improving nondifferentiable policies using model-free deep reinforcement learning. RPL thrives in complex robotic manipulation tasks where good but imperfect controllers are available. In these tasks, reinforcement learning from scratch remains data-inefficient or intractable, but learning a *residual* on top of the initial controller can yield substantial improvement. We study RPL in five challenging MuJoCo tasks involving partial observability, sensor noise, model misspecification, and controller miscalibration. By combining learning with control algorithms, RPL can perform long-horizon, sparse-reward tasks for which reinforcement learning alone fails. Moreover, we find that RPL consistently and substantially improves on the initial controllers. We argue that RPL is a promising approach for combining the complementary strengths of deep reinforcement learning and robotic control, pushing the boundaries of what either can achieve independently.

## arXiv
https://arxiv.org/abs/1812.06298

## Website
https://k-r-allen.github.io/residual-policy-learning/

## Install
Visit https://github.com/openai/baselines and install their prerequisites. (We have tested with `tensorflow==1.10.1`.)

Then:
```
pip install -r requirements.txt
```

## Usage Examples
TODO

